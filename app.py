# app.py
# Simple online predictor: loads best_model.pkl from repo, no uploads needed.
# Features: VP, MV (categorical), and a set of numeric ICU variables.
# Output: predicted probability or prediction, plus SHAP explanation.

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Basic settings
# ----------------------------
st.set_page_config(page_title="Sepsis Risk Predictor", layout="wide")
st.title("🧠 Sepsis Risk Predictor")

MODEL_PATH = Path("best_model.pkl")  # put your trained model file in repo root

# ----------------------------
# Feature schema
# Edit here if your feature names or ranges differ
# ----------------------------
FEATURE_SPECS = {
    # categorical 0/1
    "VP": {
        "type": "cat",
        "labels": {0: "No vasopressor", 1: "Vasopressor used"},
        "default": 0,
    },
    "MV": {
        "type": "cat",
        "labels": {0: "No mechanical ventilation", 1: "On mechanical ventilation"},
        "default": 0,
    },

    # numeric inputs: min, max, default, step
    "APACHEII": {"type": "num", "min": 0.0, "max": 71.0, "default": 18.0, "step": 1.0},
    "SOFA":     {"type": "num", "min": 0.0, "max": 24.0, "default": 5.0,  "step": 1.0},
    "GCS":      {"type": "num", "min": 3.0, "max": 15.0, "default": 12.0, "step": 1.0},
    "CCI":      {"type": "num", "min": 0.0, "max": 25.0, "default": 2.0,  "step": 1.0},
    "MAP":      {"type": "num", "min": 30.0, "max": 200.0,"default": 75.0, "step": 1.0},   # mmHg
    "Temp":     {"type": "num", "min": 30.0, "max": 43.0, "default": 37.8, "step": 0.1},   # °C
    "RR":       {"type": "num", "min": 5.0,  "max": 60.0, "default": 20.0, "step": 1.0},   # /min
    "Calcium":  {"type": "num", "min": 0.5,  "max": 3.0,  "default": 2.2,  "step": 0.1},   # mmol/L
    "Sodium":   {"type": "num", "min": 110.0,"max": 170.0,"default": 138.0,"step": 1.0},   # mmol/L
    "Glucose":  {"type": "num", "min": 2.0,  "max": 33.0, "default": 8.5,  "step": 0.1},   # mmol/L
    "Creatinine":{"type": "num","min": 20.0, "max": 1200.0,"default": 90.0,"step": 5.0},   # µmol/L
    "HB":       {"type": "num", "min": 40.0, "max": 200.0,"default": 120.0,"step": 1.0},   # g/L
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
    "Calcium": "Calcium (mmol/L)",
    "Sodium": "Sodium (mmol/L)",
    "Glucose": "Glucose (mmol/L)",
    "Creatinine": "Creatinine (µmol/L)",
    "HB": "Hemoglobin (g/L)",
}

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file best_model.pkl not found in repository root.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Input parameters")

user_inputs = {}
for feat in FEATURE_ORDER:
    spec = FEATURE_SPECS[feat]
    label = DISPLAY_NAME.get(feat, feat)
    if spec["type"] == "cat":
        labels = list(spec["labels"].values())
        keys = list(spec["labels"].keys())
        default_idx = keys.index(spec.get("default", keys[0]))
        sel = st.sidebar.selectbox(label, labels, index=default_idx)
        # map back to code
        inv = {v: k for k, v in spec["labels"].items()}
        user_inputs[feat] = inv[sel]
    else:
        user_inputs[feat] = st.sidebar.number_input(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["default"]),
            step=float(spec["step"]),
            format="%.3f" if spec["step"] < 1 else "%.0f"
        )

predict_btn = st.sidebar.button("🔮 Predict", type="primary")

# Build dataframe in the model's expected column order
def build_input_df(d):
    row = {k: d[k] for k in FEATURE_ORDER}
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    # cast categoricals as integers 0/1
    for c in ["VP", "MV"]:
        if c in X.columns:
            X[c] = X[c].astype(int)
    return X

# Generic probability extraction
def predict_proba_or_value(m, X):
    # try classification proba
    if hasattr(m, "predict_proba"):
        p = np.asarray(m.predict_proba(X))
        if p.ndim == 2 and p.shape[1] >= 2:
            return float(p[0, 1]), "prob"
    # try decision function with logistic
    if hasattr(m, "decision_function"):
        z = float(np.ravel(m.decision_function(X))[0])
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p), "prob"
    # fallback to predict value
    yhat = float(np.ravel(m.predict(X))[0])
    return yhat, "value"

# ----------------------------
# Main area
# ----------------------------
if predict_btn:
    X = build_input_df(user_inputs)
    score, kind = predict_proba_or_value(model, X)

    st.subheader("Prediction")
    if kind == "prob":
        st.metric("Predicted probability", f"{score:.1%}")
    else:
        st.metric("Predicted value", f"{score:.4f}")

    st.caption("Model input preview")
    st.dataframe(X, use_container_width=True)

    st.divider()
    st.subheader("SHAP explanation")

    # Try tree/linear explainer first, fallback to kernel explainer with 1-row background
    shap_values = None
    base_value = None
    try:
        expl = shap.Explainer(model)
        sv = expl(X)
        # shap>=0.40 returns Explanation
        base_value = float(np.ravel(sv.base_values)[0])
        shap_values = np.ravel(sv.values)
    except Exception:
        try:
            # background is a single baseline row built from defaults
            defaults = {k: v.get("default", 0) if v["type"] == "num" else v.get("default", 0)
                        for k, v in FEATURE_SPECS.items()}
            X_bg = pd.DataFrame([defaults], columns=FEATURE_ORDER)
            kernel = shap.KernelExplainer(lambda x: np.array([predict_proba_or_value(model, pd.DataFrame(x, columns=FEATURE_ORDER))[0] for _ in range(len(x))]),
                                          data=X_bg, link="logit")
            sv = kernel.shap_values(X, nsamples=200)  # light-weight
            shap_values = np.array(sv)[0] if isinstance(sv, list) else np.array(sv)
            # approximate base value from link function at background
            base_prob = float(predict_proba_or_value(model, X_bg)[0])
            eps = 1e-12
            base_value = np.log(np.clip(base_prob, eps, 1 - eps) / np.clip(1 - base_prob, eps, 1 - eps))
        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")
            shap_values = None

    if shap_values is not None:
        # Waterfall style plot for a single instance
        try:
            shap.plots.waterfall(shap.Explanation(values=shap_values,
                                                  base_values=base_value,
                                                  data=X.iloc[0].values,
                                                  feature_names=FEATURE_ORDER),
                                 show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception:
            # fallback to force plot via matplotlib
            try:
                shap.force_plot(base_value, shap_values, X.iloc[0].values,
                                feature_names=FEATURE_ORDER, matplotlib=True, show=False)
                plt.tight_layout()
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as e:
                st.info(f"Could not render SHAP plot: {e}")
else:
    st.info("Set parameters in the sidebar and click Predict.")
