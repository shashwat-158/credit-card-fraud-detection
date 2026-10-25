"""
Credit Card Fraud Detector (Streamlit).

Run from the repository root:

    streamlit run app/app.py

The app loads `model/fraud_model_bundle.joblib`, written by the notebook. The bundle is a dict:
    pipeline   - the fitted imblearn Pipeline (any resampler inside it is skipped when predicting)
    threshold  - operating threshold on the RAW model score, chosen on the validation block by
                 minimising expected cost
    calibrator - Platt scaling (logistic regression on logit(score)) that maps the raw score to a probability
    features   - ordered list of the engineered features the model expects
plus metadata (model name, imbalance strategy, review-cost assumption, ...).

Input CSV: the raw dataset columns Time, V1..V28, Amount (optional Class label). Feature engineering
(hour-of-day sin/cos, log1p(Amount)) is done here with the same code as in the notebook.
"""
from __future__ import annotations

import hashlib
import io
import json
from importlib import metadata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

try:
    import shap
except ImportError:  # scoring still works without it; only the explanations are disabled
    shap = None

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SEARCH_ROOTS = [HERE, HERE.parent]  # works for app/app.py and for an app.py at the repo root
BUNDLE_FILE = "fraud_model_bundle.joblib"
V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", *V_COLS, "Amount"]
TREE_MODELS = {"RandomForestClassifier", "XGBClassifier", "LGBMClassifier"}

MAX_EAGER_CSV_ROWS = 50_000  # larger files build the "download everything" CSV only on request

RULE_SAVED = "Saved threshold (cost-optimised)"
RULE_CUSTOM = "Custom score threshold"
RULE_BUDGET = "Alert budget (review top X %)"

FIGURES = [  # (file in images/, caption)
    ("test_evaluation.png", "Out-of-time test evaluation: confusion matrix, ROC, precision-recall"),
    ("pr_curves_validation.png", "Precision-recall curves of all candidate models (validation block)"),
    ("cost_vs_threshold.png", "Expected cost vs decision threshold"),
    ("calibration_reliability.png", "Calibration: raw score vs Platt-calibrated probability"),
    ("ablation_imbalance_strategy.png", "Imbalance-handling ablation on time-ordered folds"),
    ("shap_summary.png", "SHAP summary: all test frauds + 500 random legit transactions"),
]

LIMITATIONS = """
- **Short horizon:** the data covers about two days, so drift (seasonality, new fraud tactics) cannot be assessed.
- **No identifiers:** no card, customer or merchant ID, so velocity / history features are impossible.
- **Few frauds:** the test block holds only a few dozen, so metrics carry wide confidence intervals.
- **Assumed costs:** the threshold uses an assumed review cost per alert and approximates fraud loss by the
  transaction amount.
- **Anonymised features:** V1-V28 are PCA components, so explanations describe the model's reliance on
  components, not business reasons. Importance is not causation.
- **Demonstration only:** this is a portfolio project, not a production fraud system.
"""


# --------------------------------------------------------------------------------------
# Pure helpers (same logic as the notebook)
# --------------------------------------------------------------------------------------
def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw columns (Time, V1..V28, Amount) -> model features. Identical to the notebook's make_features()."""
    out = raw[V_COLS].copy()
    out["log_amount"] = np.log1p(raw["Amount"])
    tod = (raw["Time"] % 86400) / 86400  # position inside the 24 h cycle, 0..1
    out["hour_sin"] = np.sin(2 * np.pi * tod)
    out["hour_cos"] = np.cos(2 * np.pi * tod)
    return out


def logit(p, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def cost_curve(y, s, amt, review_cost: float):
    """Expected cost at every possible threshold: each alert costs `review_cost`, each missed fraud its Amount."""
    y, s, amt = np.asarray(y), np.asarray(s), np.asarray(amt)
    order = np.argsort(-s, kind="stable")
    y, s, amt = y[order], s[order], amt[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]  # last row of each distinct score (ties not split)
    caught_value = np.cumsum(amt * y)[last]
    costs = review_cost * (last + 1) + (amt * y).sum() - caught_value
    return s[last], costs  # flag when score >= threshold


def confusion_metrics(y, flagged, amount, review_cost: float) -> dict:
    y = np.asarray(y)
    flagged = np.asarray(flagged, dtype=bool)
    amount = np.asarray(amount, dtype=float)
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    fraud_value = float((amount * y).sum())
    missed_value = float((amount * y * ~flagged).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "value_caught_pct": 100 * (fraud_value - missed_value) / fraud_value if fraud_value else float("nan"),
        "cost": review_cost * (tp + fp) + missed_value,
        "no_model_cost": fraud_value,
    }


def find_path(*candidates: str) -> Path | None:
    for root in SEARCH_ROOTS:
        for rel in candidates:
            path = root / rel
            if path.exists():
                return path
    return None


def dist_version(*names: str) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


LIBS = {
    "numpy": ("numpy",), "pandas": ("pandas",), "scikit-learn": ("scikit-learn",),
    "imbalanced-learn": ("imbalanced-learn",), "xgboost": ("xgboost", "xgboost-cpu"),
    "lightgbm": ("lightgbm",), "shap": ("shap",),
}


def version_mismatches(trained: dict) -> list[str]:
    """Libraries whose version differs from the training environment (matters for unpickling the model)."""
    out = []
    for lib, names in LIBS.items():
        was, now = trained.get(lib), dist_version(*names)
        if was and now and was != now:
            out.append(f"{lib}: trained with {was}, running {now}")
    return out


def pick_sample(flagged: np.ndarray, labels, n: int, seed: int = 42) -> np.ndarray:
    """Rows for the global explanation: true frauds (or flagged rows if unlabelled) + random others."""
    rng = np.random.default_rng(seed)
    priority = np.flatnonzero(labels == 1) if labels is not None else np.flatnonzero(flagged)
    rest = np.setdiff1d(np.arange(len(flagged)), priority)
    take_p = priority if len(priority) <= n // 2 else rng.choice(priority, n // 2, replace=False)
    take_r = rng.choice(rest, min(len(rest), n - len(take_p)), replace=False)
    return np.sort(np.r_[take_p, take_r]).astype(int)


def driver_lines(values_row, data_row, names, unit: str, k: int = 3) -> str:
    order = np.argsort(values_row)
    lines = []
    for i in order[::-1][:k]:
        if values_row[i] > 0:
            lines.append(f"- `{names[i]}` = {data_row[i]:.3g} pushes the score **up** ({values_row[i]:+.4f} {unit})")
    for i in order[:k]:
        if values_row[i] < 0:
            lines.append(f"- `{names[i]}` = {data_row[i]:.3g} pushes the score **down** ({values_row[i]:+.4f} {unit})")
    return "\n".join(lines) or "No feature moves the score noticeably."


def fmt(value, pattern: str = "{:.3f}") -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else pattern.format(value)


# --------------------------------------------------------------------------------------
# Cached loaders / heavy computations
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model bundle...")
def _read_bundle(path: str, mtime: float):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "threshold", "calibrator", "features"} <= set(bundle):
        raise ValueError(
            f"{Path(path).name} is not a model bundle. Re-create it with the notebook's 'Saving the Model' cell: it "
            "must contain pipeline, threshold, calibrator and features."
        )
    bundle["path"] = path
    bundle["cache_key"] = f"{path}|{mtime}"  # every cache below includes this, so a replaced bundle is never mixed up
    return bundle


def load_bundle():
    """Load the bundle; the cache is keyed on path + modification time, so retraining takes effect immediately."""
    path = find_path(f"model/{BUNDLE_FILE}", BUNDLE_FILE)
    return None if path is None else _read_bundle(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _read_metrics(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_metrics():
    path = find_path("model/metrics.json", "metrics.json")
    return None if path is None else _read_metrics(str(path), path.stat().st_mtime)


@st.cache_resource(max_entries=2, show_spinner="Reading and scoring transactions...")
def prepare(file_bytes: bytes, bundle_key: str, _bundle: dict) -> dict:
    """Parse the CSV, drop unusable rows, engineer features and score. Cached on the file contents + model, so
    moving the threshold slider never re-scores. The returned dict is shared, so callers must not mutate it."""
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # unreadable / not a CSV
        return {"error": f"Could not read the file as CSV: {exc}"}

    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        return {"error": "Missing required column(s): " + ", ".join(missing)
                         + ". Expected the raw dataset columns Time, V1-V28, Amount (optional: Class)."}

    data = raw.copy()
    data[RAW_COLS] = data[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(data[RAW_COLS]).all(axis=1) & (data["Amount"] >= 0)
    notes = []
    if (~valid).any():
        notes.append(f"{int((~valid).sum()):,} row(s) with missing, non-numeric or negative values were excluded.")
    data = data[valid].copy()
    if data.empty:
        return {"error": "No usable rows left after removing missing / invalid values."}

    labels = None
    if "Class" in data.columns:
        cls = pd.to_numeric(data["Class"], errors="coerce")
        if cls.isin([0, 1]).all():
            labels = cls.astype(int).to_numpy()
            data["Class"] = labels
        else:
            notes.append("Column 'Class' is not strictly 0/1, so it was ignored.")

    X = make_features(data)[_bundle["features"]]
    score = _bundle["pipeline"].predict_proba(X)[:, 1]
    prob = _bundle["calibrator"].predict_proba(logit(score).reshape(-1, 1))[:, 1]
    return {"data": data, "X": X, "score": score, "prob": prob, "labels": labels, "notes": notes,
            "digest": hashlib.md5(file_bytes).hexdigest()}


@st.cache_data(max_entries=4, show_spinner=False)
def to_csv_bytes(key: tuple, _df: pd.DataFrame) -> bytes:
    return _df.to_csv(index=False).encode("utf-8")


@st.cache_resource(show_spinner=False)
def get_tree_explainer(bundle_key: str, _model):
    return shap.TreeExplainer(_model)


def explain(bundle: dict, X: pd.DataFrame):
    """SHAP values of the FRAUD class for the rows of X -> (values [n, f], base values [n], unit) or None."""
    if shap is None:
        return None
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    kind = type(model).__name__
    if kind in TREE_MODELS:
        ex = get_tree_explainer(bundle["cache_key"], model)(X, check_additivity=False)
        values, base = np.asarray(ex.values), np.asarray(ex.base_values)
        if values.ndim == 3:  # random forest: one block per class, keep "fraud"
            values, base = values[:, :, 1], base[:, 1]
        if base.ndim == 0:
            base = np.full(len(X), float(base))
        return values, base, ("probability" if kind == "RandomForestClassifier" else "log-odds")
    if kind == "LogisticRegression":  # exact linear SHAP: the scaler centres the training data at 0
        Z = X
        for _, step in pipe.steps[:-1]:
            if hasattr(step, "transform"):  # scaler; resamplers have no transform and are skipped
                Z = step.transform(Z)
        return np.asarray(Z) * model.coef_[0], np.full(len(X), float(model.intercept_[0])), "log-odds"
    return None


@st.cache_data(max_entries=2, show_spinner=False)
def global_shap(key: tuple, _bundle: dict, _X: pd.DataFrame, _positions: np.ndarray):
    X_s = _X.iloc[_positions]
    out = explain(_bundle, X_s)
    return None if out is None else (*out, X_s)


# --------------------------------------------------------------------------------------
# UI pieces
# --------------------------------------------------------------------------------------
def sidebar_model_info(bundle: dict, metrics: dict | None) -> None:
    st.sidebar.subheader("Model")
    model_name = bundle.get("model_name") or type(bundle["pipeline"].named_steps["model"]).__name__
    st.sidebar.markdown(
        f"**{model_name}**  \nimbalance handling: `{bundle.get('imbalance_strategy', 'n/a')}`  \n"
        f"{bundle.get('trained_on', '')}"
    )
    thr = float(bundle["threshold"])
    p_thr = bundle["calibrator"].predict_proba(logit([thr]).reshape(-1, 1))[0, 1]
    st.sidebar.caption(f"Saved threshold {thr:.3f} on the raw score, about {p_thr:.1%} calibrated fraud probability.")
    if metrics and metrics.get("versions"):
        mismatches = version_mismatches(metrics["versions"])
        if mismatches:
            st.sidebar.warning(
                "Library versions differ from the training environment (pickled models can break or change):\n\n- "
                + "\n- ".join(mismatches)
            )


def input_source():
    uploaded = st.file_uploader(
        "Upload a CSV of transactions", type="csv",
        help="Raw dataset columns: Time, V1-V28, Amount. An optional Class column (0/1) enables evaluation.",
    )
    sample = find_path("app/sample.csv", "sample.csv", "data/sample.csv")
    use_sample = False
    if sample is not None:
        use_sample = st.checkbox("Use the bundled sample file instead", value=False)
    if use_sample:
        return sample.read_bytes(), sample.name
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    return None


def decision_rule(bundle: dict, score: np.ndarray):
    saved = float(bundle["threshold"])
    st.sidebar.subheader("Decision rule")
    mode = st.sidebar.radio("Flag a transaction when...", [RULE_SAVED, RULE_CUSTOM, RULE_BUDGET],
                            label_visibility="collapsed")
    if mode == RULE_SAVED:
        thr = saved
    elif mode == RULE_CUSTOM:
        thr = st.sidebar.slider("Raw model-score threshold", 0.0, 1.0, float(np.clip(round(saved, 3), 0, 1)),
                                0.005, format="%.3f")
    else:
        pct = st.sidebar.slider("Share of transactions to review (%)", 0.01, 5.0, 0.10, 0.01, format="%.2f")
        k = min(len(score), max(1, int(np.ceil(pct / 100 * len(score)))))
        thr = float(np.partition(score, -k)[-k])
    return thr, mode


def render_kpis(n: int, flagged: np.ndarray, amount: np.ndarray, review_cost: float, labels) -> None:
    n_flag = int(flagged.sum())
    cols = st.columns(4)
    cols[0].metric("Transactions", f"{n:,}")
    cols[1].metric("Flagged", f"{n_flag:,}", f"{100 * n_flag / n:.2f}% of transactions", delta_color="off")
    cols[2].metric("Amount flagged", f"{amount[flagged].sum():,.0f}")
    cols[3].metric("Est. review cost", f"{n_flag * review_cost:,.0f}", help="flagged transactions x review cost")
    if labels is None:
        return
    m = confusion_metrics(labels, flagged, amount, review_cost)
    st.caption("The file has a Class column, so the decision rule can be evaluated against it:")
    c = st.columns(4)
    c[0].metric("Frauds caught", f"{m['tp']} / {m['tp'] + m['fn']}", f"{m['fn']} missed", delta_color="off")
    c[1].metric("False alarms", f"{m['fp']:,}")
    c[2].metric("Precision / recall", f"{fmt(m['precision'], '{:.2f}')} / {fmt(m['recall'], '{:.2f}')}")
    c[3].metric("Fraud value caught", fmt(m["value_caught_pct"], "{:.1f}%"))
    if m["no_model_cost"] > 0:
        saving = 100 * (1 - m["cost"] / m["no_model_cost"])
        st.caption(f"Expected cost {m['cost']:,.0f} vs {m['no_model_cost']:,.0f} when flagging nothing "
                   f"({saving:.0f}% lower) at {review_cost:g} per alert.")


def tab_results(results: pd.DataFrame, score: np.ndarray, thr: float, labelled: bool, key: tuple) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.hist(score, bins=50, color="#3498db")
    ax.set_yscale("log")
    ax.axvline(thr, color="#e74c3c", linestyle="--", label=f"threshold {thr:.3f}")
    ax.set_xlabel("Raw fraud score")
    ax.set_ylabel("Transactions (log)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    c1, c2, c3, c4 = st.columns(4)
    only_flagged = c1.checkbox("Only flagged", value=True)
    sort_by = c2.selectbox("Sort by", ["fraud_score", "expected_loss", "Amount"],
                           help="expected_loss = calibrated probability x Amount")
    n_show = c3.slider("Rows shown", 10, 1000, 100, 10)
    all_cols = c4.checkbox("Show all columns", value=False)

    compact = ["row", "Time", "Amount", "fraud_score", "fraud_probability", "expected_loss", "flagged"]
    if labelled:
        compact += ["Class", "outcome"]
    view = results[results["flagged"]] if only_flagged else results
    view = view.sort_values(sort_by, ascending=False).head(n_show)
    st.dataframe(
        view if all_cols else view[compact], hide_index=True,
        column_config={
            "fraud_score": st.column_config.ProgressColumn("fraud_score", min_value=0.0, max_value=1.0, format="%.3f"),
            "fraud_probability": st.column_config.NumberColumn("fraud_probability", format="%.4f"),
            "expected_loss": st.column_config.NumberColumn("expected_loss", format="%.2f"),
        },
    )
    st.caption(f"Showing {len(view):,} of {int(results['flagged'].sum()) if only_flagged else len(results):,} rows. "
               "`fraud_score` is the raw model score used for the decision; `fraud_probability` is the calibrated "
               "estimate.")
    d1, d2 = st.columns(2)
    if len(results) <= MAX_EAGER_CSV_ROWS or d1.checkbox("Prepare the full scored CSV (large file)"):
        d1.download_button("Download all scored transactions (CSV)", to_csv_bytes(key + ("all",), results),
                           "scored_transactions.csv", "text/csv")
    d2.download_button("Download flagged only (CSV)", to_csv_bytes(key + ("flagged",), results[results["flagged"]]),
                       "flagged_transactions.csv", "text/csv")


def tab_explain(bundle: dict, results: pd.DataFrame, X: pd.DataFrame, labelled: bool) -> None:
    if shap is None:
        st.info("Install `shap` (`pip install shap`) to enable explanations.")
        return
    top = results["fraud_score"].nlargest(300).index.tolist()

    def label(i):
        extra = f" | actual class {int(results.at[i, 'Class'])}" if labelled else ""
        return f"row {i} | score {results.at[i, 'fraud_score']:.3f} | amount {results.at[i, 'Amount']:.2f}{extra}"

    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Transaction (top 300 by score)", top, format_func=label)
    lookup = c2.number_input("...or any row number", min_value=-1, value=-1, step=1,
                             help="-1 = use the selection on the left")
    idx = int(lookup) if lookup >= 0 and lookup in results.index else choice
    if lookup >= 0 and lookup not in results.index:
        st.warning(f"Row {int(lookup)} is not in the scored data (rows with invalid values are excluded).")

    row = results.loc[idx]
    m = st.columns(3)
    m[0].metric("Raw fraud score", f"{row['fraud_score']:.3f}")
    m[1].metric("Calibrated probability", f"{row['fraud_probability']:.2%}")
    m[2].metric("Decision", "FLAGGED" if row["flagged"] else "not flagged")

    out = explain(bundle, X.loc[[idx]])
    if out is None:
        st.info("Local explanations are not available for this model type.")
        return
    values, base, unit = out
    names = list(X.columns)
    data_row = X.loc[idx].to_numpy()
    expl = shap.Explanation(values=values[0], base_values=float(base[0]), data=data_row, feature_names=names)
    plt.close("all")  # SHAP draws on the current figure, so start clean
    shap.plots.waterfall(expl, max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Bars show each feature's push on the model output, starting from the average output "
               f"(E[f(x)] = {base[0]:.4f}) and ending at this transaction's score. Units: {unit}. "
               "V1-V28 are PCA components, so they have no business meaning of their own.")
    st.markdown(driver_lines(values[0], data_row, names, unit))


def tab_global(bundle: dict, X: pd.DataFrame, flagged: np.ndarray, labels, key: tuple) -> None:
    if shap is None:
        st.info("Install `shap` (`pip install shap`) to enable explanations.")
        return
    st.write("Which features does the model rely on across this file? SHAP is computed on a sample "
             "(true frauds if the file is labelled, otherwise flagged rows, plus random other transactions).")
    n = st.slider("Sample size", 50, 1000, 300, 50, help="Larger samples are slower, especially for forests.")
    gkey = key + (n,)
    if st.button("Compute global explanation"):
        st.session_state["global_key"] = gkey
    if st.session_state.get("global_key") != gkey:
        return
    with st.spinner("Computing SHAP values..."):
        res = global_shap(gkey, bundle, X, pick_sample(flagged, labels, n))
    if res is None:
        st.info("Global explanations are not available for this model type.")
        return
    values, _, unit, X_s = res
    plt.close("all")
    shap.summary_plot(values, X_s, max_display=15, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

    imp = pd.Series(np.abs(values).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
    rows = []
    for f in imp.index[:8]:
        r = np.corrcoef(X_s[f].to_numpy(), values[:, X_s.columns.get_loc(f)])[0, 1]
        effect = ("higher values raise the score" if r > 0.2 else
                  "lower values raise the score" if r < -0.2 else "mixed effect")
        rows.append({"feature": f, "mean |SHAP|": round(float(imp[f]), 5), "effect": effect})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Units: {unit}. The ranking describes what the model relies on in this sample, not what causes fraud; "
               "the sample over-represents frauds / flagged rows.")


def tab_threshold(score, amount, labels, thr, saved, review_cost) -> None:
    if labels is None:
        st.info("Add a `Class` column (0/1) to the file to analyse thresholds against known outcomes.")
        return
    thr_grid, cost = cost_curve(labels, score, amount, review_cost)
    best = float(thr_grid[np.argmin(cost)])
    no_model = float((amount * labels).sum())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(thr_grid, cost, color="#e74c3c")
    ax.axhline(no_model, color="gray", linestyle="--", label="flag nothing")
    ax.axvline(thr, color="black", linestyle=":", label=f"current ({thr:.3f})")
    ax.axvline(best, color="green", linestyle=":", label=f"best on this file ({best:.3f})")
    ax.set_xlabel("Raw-score threshold")
    ax.set_ylabel("Expected cost")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    rows = []
    for name, t in [("Current rule", thr), ("Saved threshold", saved), ("Best on this file", best)]:
        m = confusion_metrics(labels, score >= t, amount, review_cost)
        rows.append({"rule": name, "threshold": round(t, 4), "flagged": m["tp"] + m["fp"], "caught": m["tp"],
                     "missed": m["fn"], "precision": round(m["precision"], 3), "recall": round(m["recall"], 3),
                     "expected_cost": round(m["cost"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("The 'best on this file' threshold is tuned on the data it is scored on, so it is optimistic. "
               "Use it to see how flat or steep the cost curve is, not as a replacement for the validated threshold.")


def render_model_card(bundle: dict, metrics: dict | None) -> None:
    thr = float(bundle["threshold"])
    st.markdown(
        f"**Model:** {bundle.get('model_name', 'n/a')}  \n"
        f"**Imbalance handling:** `{bundle.get('imbalance_strategy', 'n/a')}`  \n"
        f"**Training window:** {bundle.get('trained_on', 'n/a')}  \n"
        f"**Saved threshold (raw score):** {thr:.3f}  \n"
        f"**Review-cost assumption:** {bundle.get('review_cost_assumption', 'n/a')} per alert; a missed fraud costs "
        "its transaction amount"
    )
    if metrics:
        tb = metrics.get("test_block", {})
        st.subheader("Out-of-time test performance")
        if tb:
            st.caption(f"{tb.get('transactions', 0):,} test transactions, {tb.get('frauds', 0)} frauds "
                       "(estimate with 95% bootstrap interval)")
        names = {"pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "precision": "Precision", "recall": "Recall", "f1": "F1",
                 "mcc": "MCC", "fraud_value_caught_%": "Fraud value caught (%)"}
        if metrics.get("test_metrics"):
            tm = pd.DataFrame(metrics["test_metrics"]).T.rename(index=names).round(3)
            st.dataframe(tm.rename(columns={"estimate": "Estimate", "ci_low": "95% CI low", "ci_high": "95% CI high"}))
        if metrics.get("protocol_comparison"):
            st.markdown("**Random split vs out-of-time split** (same model, same threshold)")
            st.dataframe(pd.DataFrame(metrics["protocol_comparison"]).set_index("protocol").round(3))
        if metrics.get("model_ranking"):
            st.markdown("**Model comparison on the validation block** (baselines included)")
            st.dataframe(pd.DataFrame(metrics["model_ranking"]).set_index("model").round(4))
    else:
        st.info("Run the notebook's export cell to add test metrics (`model/metrics.json`) to this page.")

    images = find_path("images")
    shown = [(images / f, c) for f, c in FIGURES if images is not None and (images / f).exists()]
    if shown:
        st.subheader("Figures from the notebook")
        for path, caption in shown:
            st.image(str(path), caption=caption)
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🛡️", layout="wide")
    st.title("🛡️ Credit Card Fraud Detector")
    st.caption("Score raw transactions, tune the decision rule, and see why a transaction was flagged.")

    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error(f"Could not load the model bundle: {exc}")
        st.stop()
    if bundle is None:
        st.error(f"`{BUNDLE_FILE}` not found. Run the notebook (including its final export cell) to create it in "
                 "`model/`, then restart the app.")
        st.stop()

    metrics = load_metrics()
    sidebar_model_info(bundle, metrics)

    source = input_source()
    if source is None:
        st.info("Upload a CSV of raw transactions (columns `Time`, `V1`-`V28`, `Amount`; optional `Class`) "
                "or tick the bundled sample to try the app.")
        render_model_card(bundle, metrics)
        return

    file_bytes, file_name = source
    prepared = prepare(file_bytes, bundle["cache_key"], bundle)
    if "error" in prepared:
        st.error(prepared["error"])
        st.stop()
    for note in prepared["notes"]:
        st.warning(note)

    data, X, score, prob, labels = (prepared[k] for k in ("data", "X", "score", "prob", "labels"))
    amount = data["Amount"].to_numpy()
    labelled = labels is not None

    thr, rule = decision_rule(bundle, score)
    review_cost = st.sidebar.number_input("Review cost per alert", min_value=0.0, step=1.0,
                                          value=float(bundle.get("review_cost_assumption", 5.0)),
                                          help="Used for the review-cost and expected-cost figures.")
    flagged = score >= thr

    results = data.copy()
    results.insert(0, "row", data.index)
    results["fraud_score"] = score
    results["fraud_probability"] = prob
    results["expected_loss"] = prob * amount
    results["flagged"] = flagged
    if labelled:
        results["outcome"] = np.select(
            [flagged & (labels == 1), flagged & (labels == 0), ~flagged & (labels == 1)],
            ["true positive", "false alarm", "missed fraud"], default="true negative")

    st.subheader(f"{file_name}: {int(flagged.sum()):,} flagged out of {len(results):,}")
    st.caption(f"Rule: {rule}. Flagging raw scores >= {thr:.3f}.")
    render_kpis(len(results), flagged, amount, review_cost, labels)

    key = (bundle["cache_key"], prepared["digest"], round(thr, 8))
    t1, t2, t3, t4, t5 = st.tabs(["Results", "Explain a transaction", "Global drivers",
                                  "Threshold analysis", "Model card"])
    with t1:
        tab_results(results, score, thr, labelled, key)
    with t2:
        tab_explain(bundle, results, X, labelled)
    with t3:
        tab_global(bundle, X, flagged, labels, key[:2] if labelled else key)
    with t4:
        tab_threshold(score, amount, labels, thr, float(bundle["threshold"]), review_cost)
    with t5:
        render_model_card(bundle, metrics)


if __name__ == "__main__":
    main()