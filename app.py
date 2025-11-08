# app.py
# ARDS Risk Predictor for Severe TBI Patients
# Loads best_model.pkl from repo root; no uploads needed.
# Features used at training time:
# VP, MV, APACHEII, SOFA, GCS, CCI, MAP, Temp, RR, Calcium, Sodium, Glucose, Creatinine, HB

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ARDS Risk Predictor for Severe TBI Patients", layout="wide")
st.title("🧠 ARDS Risk Predictor for Severe TBI Patients")

MODEL_PATH = Path("best_model.pkl")  # your trained model here

# ----------------------------
# Feature schema (matches your training list)
# ----------------------------
FEATURE_SPECS = {
    "VP":         {"type": "cat", "labels": {0: "No vasopressor", 1: "Vasopressor used"}, "default": 0},
    "MV":         {"type": "cat", "labels": {0: "No mechanical ventilation", 1: "Mechanical ventilation"}, "default": 0},
    "APACHEII":   {"type": "num", "min": 0.0,  "max": 71.0,  "default": 18.0, "step": 1.0},
    "SOFA":       {"type": "num", "min": 0.0,  "max": 24.0,  "default": 6.0,  "step": 1.0},
    "GCS":        {"type": "num", "min": 3.0,  "max": 15.0,  "default": 10.0, "step": 1.0},
    "CCI":        {"type": "num", "min": 0.0,  "max": 20.0,  "default": 2.0,  "step": 1.0},
    "MAP":        {"type": "num", "min": 30.0, "max": 200.0, "default": 80.0, "step": 1.0},   # mmHg
    "Temp":       {"type": "num", "min": 30.0, "max": 43.0,  "default": 37.5, "step": 0.1},   # °C
    "RR":         {"type": "num", "min": 5.0,  "max": 60.0,  "default": 20.0, "step": 1.0},   # /min
    "Calcium":    {"type": "num", "min": 0.5,  "max": 3.0,   "default": 2.2,  "step": 0.1},   # mmol/L
    "Sodium":     {"type": "num", "min": 110.0,"max": 170.0, "default": 138.0,"step": 1.0},   # mmol/L
    "Glucose":    {"type": "num", "min": 2.0,  "max": 33.0,  "default": 8.5,  "step": 0.1},   # mmol/L
    "Creatinine": {"type": "num", "min": 20.0, "max": 1200.0,"default": 90.0, "step": 5.0},   # µmol/L
    "HB":         {"type": "num", "min": 40.0, "max": 200.0, "default": 120.0,"step": 1.0},   # g/L
}
FEATURE_ORDER = list(FEATURE_SPECS.keys())

DISPLAY_NAME = {
    "VP": "Vasopressor use",
    "MV": "Mechanical ventilation",
    "APACHEII": "APACHE-II (score)",
    "SOFA": "SOFA (score)",
    "GCS": "GCS (score)",
    "CCI": "Charlson comorbidity index",
    "MAP": "Mean arterial pressure (mmHg)",
    "Temp": "Temperature (°C)",
    "RR": "Respiratory rate (/min)",
    "Calcium": "Serum calcium (mmol/L)",
    "Sodium": "Serum sodium (mmol/L)",
    "Glucose": "Blood glucose (mmol/L)",
    "Creatinine": "Creatinine (µmol/L)",
    "HB": "Hemoglobin (g/L)",
}

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file 'best_model.pkl' not found in repository root.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------
# Align input to model's training schema
# ----------------------------
def align_to_model_features(model, X, cast_cat="int"):
    """
    Reorder/complete/drop columns to match model.feature_names_in_.
    - cast_cat: "int" to keep VP/MV as integers; "str" if your pipeline expects strings.
    """
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is None and hasattr(model, "named_steps"):
        try:
            feat_in = getattr(model.named_steps.get("clf", None), "feature_names_in_", None)
        except Exception:
            feat_in = None

    if feat_in is None:
        # No strict schema saved; just ensure types
        Z = X.copy()
        if cast_cat == "int":
            for c in ["VP", "MV"]:
                if c in Z.columns:
                    Z[c] = Z[c].astype(int)
        else:
            for c in ["VP", "MV"]:
                if c in Z.columns:
                    Z[c] = Z[c].astype("Int64").astype(str)
        return Z

    feat_in = list(feat_in)
    lower_map = {c.lower(): c for c in X.columns}
    aligned = {}
    for col in feat_in:
        if col in X.columns:
            aligned[col] = X[col]
        elif col.lower() in lower_map:
            aligned[col] = X[lower_map[col.lower()]]
        else:
            aligned[col] = 0.0  # missing -> 0

    Z = pd.DataFrame(aligned, columns=feat_in)

    if cast_cat == "int":
        for c in ["VP", "MV"]:
            if c in Z.columns:
                try:
                    Z[c] = Z[c].astype(int)
                except Exception:
                    pass
    else:
        for c in ["VP", "MV"]:
            if c in Z.columns:
                Z[c] = Z[c].astype("Int64").astype(str)
    return Z

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("🩺 Patient Parameters")
inputs = {}
for feat in FEATURE_ORDER:
    spec = FEATURE_SPECS[feat]
    label = DISPLAY_NAME.get(feat, feat)
    if spec["type"] == "cat":
        labels = list(spec["labels"].values())
        default_idx = list(spec["labels"].keys()).index(spec["default"])
        sel = st.sidebar.selectbox(label, labels, index=default_idx)
        inputs[feat] = {v: k for k, v in spec["labels"].items()}[sel]
    else:
        inputs[feat] = st.sidebar.number_input(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
            step=float(spec["step"]),
        )

predict_btn = st.sidebar.button("🔮 Predict ARDS Risk", type="primary")

def build_input_df(d):
    return pd.DataFrame([{k: d[k] for k in FEATURE_ORDER}], columns=FEATURE_ORDER)

def predict_proba_or_value(m, X):
    if hasattr(m, "predict_proba"):
        p = np.asarray(m.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(p[0, 1]), "prob"
    if hasattr(m, "decision_function"):
        z = float(np.ravel(m.decision_function(X))[0])
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p), "prob"
    yhat = float(np.ravel(m.predict(X))[0])
    return yhat, "value"

# ----------------------------
# Main
# ----------------------------
if predict_btn:
    X_raw = build_input_df(inputs)

    # 如果你的 Pipeline 里对 VP/MV 要求字符串类别，把 cast_cat 改成 "str"
    X = align_to_model_features(model, X_raw, cast_cat="int")

    with st.expander("Debug: feature schema"):
        st.write("Expected by model:", list(getattr(model, "feature_names_in_", [])))
        st.write("Provided (after alignment):", list(X.columns))

    score, kind = predict_proba_or_value(model, X)

    st.subheader("🎯 Prediction")
    if kind == "prob":
        st.metric("Predicted ARDS probability", f"{score:.1%}")
    else:
        st.metric("Predicted output", f"{score:.4f}")

    st.caption("Input summary")
    st.dataframe(X_raw, use_container_width=True)

    st.divider()
    st.subheader("📊 SHAP Explanation")

    shap_values = None
    base_value = None
    try:
        expl = shap.Explainer(model)
        sv = expl(X)
        base_value = float(np.ravel(sv.base_values)[0])
        shap_values = np.ravel(sv.values)
    except Exception:
        try:
            defaults = {k: v["default"] if FEATURE_SPECS[k]["type"] == "num" else FEATURE_SPECS[k]["default"]
                        for k in FEATURE_ORDER}
            X_bg_raw = pd.DataFrame([defaults], columns=FEATURE_ORDER)
            X_bg = align_to_model_features(model, X_bg_raw, cast_cat="int")
            def _pred_func(arr):
                df = pd.DataFrame(arr, columns=X.columns)
                return np.array([predict_proba_or_value(model, df.iloc[[i]])[0] for i in range(len(df))])
            kernel = shap.KernelExplainer(_pred_func, data=X_bg, link="logit")
            sv = kernel.shap_values(X, nsamples=200)
            shap_values = np.array(sv)[0] if isinstance(sv, list) else np.array(sv)
            base_prob = predict_proba_or_value(model, X_bg)[0]
            eps = 1e-12
            base_value = np.log(np.clip(base_prob, eps, 1 - eps) / np.clip(1 - base_prob, eps, 1 - eps))
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
            shap_values = None

    if shap_values is not None:
        try:
            shap.plots.waterfall(
                shap.Explanation(values=shap_values, base_values=base_value,
                                 data=X.iloc[0].values, feature_names=list(X.columns)),
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            try:
                shap.force_plot(base_value, shap_values, X.iloc[0].values,
                                feature_names=list(X.columns), matplotlib=True, show=False)
                plt.tight_layout()
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.info(f"Could not render SHAP plot: {e}")
else:
    st.info("Adjust the patient parameters on the left and click **Predict ARDS Risk**.")
