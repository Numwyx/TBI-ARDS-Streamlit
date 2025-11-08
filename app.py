# app.py — TBI-ARDS Risk Predictor (Full Stacking; level-1 PKLs in repo root)
# Files in repo root:
#   best_model.pkl
#   model_ada.pkl, model_cat.pkl, model_dt.pkl, model_gbm.pkl, model_knn.pkl,
#   model_lgb.pkl, model_logistic.pkl, model_mlp.pkl, model_rf.pkl, model_svm.pkl, model_xgb.pkl

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="TBI-ARDS Stacking Predictor", layout="wide")
st.title("🧠 TBI-ARDS Stacking Predictor")

# ---------- Paths & names ----------
ROOT = Path(".")
META_PATH = ROOT / "best_model.pkl"  # level-2 model
# the 11 level-1 feature names expected by the meta model
META_FEATURES_DEFAULT = ["ada","cat","dt","gbm","knn","lgb","logistic","mlp","rf","svm","xgb"]

# ---------- Clinical feature schema for UI ----------
FEATURE_SPECS = {
    "VP":         {"type": "cat", "labels": {0: "No vasopressor", 1: "Vasopressor used"}, "default": 0},
    "MV":         {"type": "cat", "labels": {0: "No mechanical ventilation", 1: "Mechanical ventilation"}, "default": 0},
    "APACHEII":   {"type": "num", "min": 0.0,  "max": 71.0,  "default": 18.0, "step": 1.0},
    "SOFA":       {"type": "num", "min": 0.0,  "max": 24.0,  "default": 6.0,  "step": 1.0},
    "GCS":        {"type": "num", "min": 3.0,  "max": 15.0,  "default": 10.0, "step": 1.0},
    "CCI":        {"type": "num", "min": 0.0,  "max": 20.0,  "default": 2.0,  "step": 1.0},
    "MAP":        {"type": "num", "min": 30.0, "max": 200.0, "default": 80.0, "step": 1.0},
    "Temp":       {"type": "num", "min": 30.0, "max": 43.0,  "default": 37.5, "step": 0.1},
    "RR":         {"type": "num", "min": 5.0,  "max": 60.0,  "default": 20.0, "step": 1.0},
    "Calcium":    {"type": "num", "min": 0.5,  "max": 3.0,   "default": 2.2,  "step": 0.1},
    "Sodium":     {"type": "num", "min": 110.0,"max": 170.0, "default": 138.0,"step": 1.0},
    "Glucose":    {"type": "num", "min": 2.0,  "max": 33.0,  "default": 8.5,  "step": 0.1},
    "Creatinine": {"type": "num", "min": 20.0, "max": 1200.0,"default": 90.0, "step": 5.0},
    "HB":         {"type": "num", "min": 40.0, "max": 200.0, "default": 120.0,"step": 1.0},
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())

# ---------- Utilities ----------
def expected_features(model):
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is None and hasattr(model, "named_steps"):
        try:
            feat_in = getattr(model.named_steps.get("clf", None), "feature_names_in_", None)
        except Exception:
            feat_in = None
    return list(feat_in) if feat_in is not None else None

def build_clinical_df(d):
    return pd.DataFrame([{k: d[k] for k in FEATURE_ORDER}], columns=FEATURE_ORDER)

def cast_categories(df, mode="int"):
    Z = df.copy()
    if mode == "int":
        for c in ["VP","MV"]:
            if c in Z.columns:
                try: Z[c] = Z[c].astype(int)
                except Exception: pass
    else:
        for c in ["VP","MV"]:
            if c in Z.columns:
                Z[c] = Z[c].astype("Int64").astype(str)
    return Z

def align_to_model_schema(model, X, defaults=None):
    """Align X to training schema of a given sklearn model using its feature_names_in_ if available."""
    feat_in = expected_features(model)
    if feat_in is None:
        return X.copy()
    X2 = X.copy()
    # case-insensitive rename
    lower_to_train = {c.lower(): c for c in feat_in}
    ren = {}
    for col in X2.columns:
        tgt = lower_to_train.get(col.lower())
        if tgt and tgt != col:
            ren[col] = tgt
    if ren:
        X2 = X2.rename(columns=ren)
    # add missing
    missing = [c for c in feat_in if c not in X2.columns]
    if missing:
        for c in missing:
            X2[c] = 0.0 if defaults is None or c not in defaults else defaults[c]
    return X2.reindex(columns=feat_in)

def predict_proba_or_value(model, X):
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(p[0,1]), "prob"
    if hasattr(model, "decision_function"):
        z = float(np.ravel(model.decision_function(X))[0])
        p = 1.0/(1.0+np.exp(-z))
        return float(p), "prob"
    yhat = float(np.ravel(model.predict(X))[0])
    return yhat, "value"

# ---------- Load models ----------
@st.cache_resource
def load_meta_model():
    if not META_PATH.exists():
        st.error("Missing meta model file: best_model.pkl")
        st.stop()
    return joblib.load(META_PATH)

@st.cache_resource
def load_level1_models_from_root(feature_names):
    """Load model_<name>.pkl for each meta feature from the repository root."""
    models, missing = {}, []
    for name in feature_names:
        fp = ROOT / f"model_{name}.pkl"
        if fp.exists():
            try:
                models[name] = joblib.load(fp)   # ← 真正读取一级模型
            except Exception as e:
                st.error(f"Failed to load {fp.name}: {e}")
                st.stop()
        else:
            missing.append(name)
    return models, missing

meta = load_meta_model()

# Decide feature names the meta model expects
exp_meta_feats = expected_features(meta)
if exp_meta_feats is None:
    exp_meta_feats = META_FEATURES_DEFAULT

level1_models, missing = load_level1_models_from_root(exp_meta_feats)
if missing:
    st.error("Missing level-1 model files in repository root:\n" +
             "\n".join([f"model_{n}.pkl" for n in missing]))
    st.stop()

# ---------- Strict level-1 prediction ----------
def run_level1_strict(model1, X_clin, name):
    """Predict with a level-1 model; try int then str categories. Fail loudly with context."""
    errors = []
    for mode in ("int", "str"):
        Z = cast_categories(X_clin, mode)
        Z = align_to_model_schema(model1, Z)
        try:
            p, _ = predict_proba_or_value(model1, Z)   # ← 真正调用一级模型预测
            return float(p), mode
        except Exception as e:
            errors.append((mode, str(e), list(Z.columns)))
    # both modes failed
    msg = [f"Level-1 model '{name}' failed in both modes:"]
    for m, err, cols in errors:
        msg.append(f"  - mode={m}, error={err}, cols={cols}")
    raise RuntimeError("\n".join(msg))

def build_meta_input(level1_dict, X_clin, feature_names):
    """Return (X_meta, modes_used)"""
    rec, modes = {}, {}
    for name in feature_names:
        if name not in level1_dict:
            raise FileNotFoundError(f"Missing level-1 model: model_{name}.pkl")
        p, mode = run_level1_strict(level1_dict[name], X_clin, name)
        rec[name] = p
        modes[name] = mode
    X_meta = pd.DataFrame([[rec[c] for c in feature_names]], columns=feature_names)
    return X_meta, modes

# ---------- Sidebar UI ----------
st.sidebar.header("Patient Parameters")
inputs = {}
for feat, spec in FEATURE_SPECS.items():
    label = {
        "MAP":"Mean arterial pressure (mmHg)",
        "RR":"Respiratory rate (/min)",
        "HB":"Hemoglobin (g/L)",
    }.get(feat, feat)
    if spec["type"] == "cat":
        labels = list(spec["labels"].values())
        idx = list(spec["labels"].keys()).index(spec["default"])
        sel = st.sidebar.selectbox(label, labels, index=idx)
        inputs[feat] = {v:k for k,v in spec["labels"].items()}[sel]
    else:
        inputs[feat] = st.sidebar.number_input(
            label, min_value=float(spec["min"]), max_value=float(spec["max"]),
            value=float(spec["default"]), step=float(spec["step"])
        )

go = st.sidebar.button("🔮 Predict ARDS Risk", type="primary")

# ---------- Main flow ----------
if go:
    X_clin_raw = build_clinical_df(inputs)
    X_clin = cast_categories(X_clin_raw, "int")  # first try int; each L1 can switch to str internally

    # 1) run all level-1 models
    X_meta, l1_modes = build_meta_input(level1_models, X_clin, exp_meta_feats)

    st.subheader("Level-1 outputs (probabilities fed into meta model)")
    st.dataframe(X_meta, use_container_width=True)
    st.caption("Casting mode per level-1: " + ", ".join([f"{k}:{v}" for k,v in l1_modes.items()]))

    # quick sanity checks
    if np.allclose(X_meta.values, 0.0):
        st.error("All level-1 outputs are zeros. Likely schema/dtype mismatch for level-1 pipelines.")
    elif np.max(np.abs(X_meta.values - X_meta.values.mean())) < 1e-6:
        st.warning("All level-1 outputs are almost identical constants. Check VP/MV dtype and preprocessing in level-1 models.")

    # 2) align to meta schema if it carries feature_names_in_
    X_meta = align_to_model_schema(meta, X_meta, defaults=None)

    # 3) meta prediction
    score, kind = predict_proba_or_value(meta, X_meta)

    st.subheader("🎯 Stacking Prediction")
    if kind == "prob":
        st.metric("Predicted ARDS probability", f"{score:.1%}")
    else:
        st.metric("Predicted output", f"{score:.4f}")

    st.divider()
    st.subheader("📊 SHAP Explanation (meta model)")

    shap_values = None
    base = None
    try:
        expl = shap.Explainer(meta)
        sv = expl(X_meta)
        base = float(np.ravel(sv.base_values)[0])
        shap_values = np.ravel(sv.values)
    except Exception as e:
        st.info(f"SHAP not available: {e}")

    if shap_values is not None:
        try:
            shap.plots.waterfall(
                shap.Explanation(values=shap_values, base_values=base,
                                 data=X_meta.iloc[0].values,
                                 feature_names=list(X_meta.columns)),
                show=False
            )
            plt.tight_layout(); st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            try:
                shap.force_plot(base, shap_values, X_meta.iloc[0].values,
                                feature_names=list(X_meta.columns), matplotlib=True, show=False)
                plt.tight_layout(); st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.info(f"Could not render SHAP plot: {e}")

    with st.expander("Debug"):
        st.write("Meta expects:", exp_meta_feats)
        st.write("Loaded level-1:", sorted(level1_models.keys()))
        st.write("Clinical columns:", list(X_clin_raw.columns))
else:
    st.info("Set patient parameters on the left and click Predict ARDS Risk.")
