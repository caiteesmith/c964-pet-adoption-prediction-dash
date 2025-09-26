from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Page setup ----------
st.set_page_config(page_title="Pet Adoption Prediction Dashboard", layout="wide")
st.title("🐾 Pet Shelter Adoption Prediction Dashboard 🐾")
st.caption(
    """
    Welcome to the Pet Shelter Adoption Prediction Dashboard!

    This app helps animal shelters make better decisions about space, staffing,
    and marketing by showing adoption trends and predicting the likelihood that a pet
    will be adopted quickly.

    You can explore the sample data that ships with the app or upload your own shelter
    records. The dashboard gives you:
    * **Key data insights** like typical adoption fees and the age mix of pets
    * **Visual relationships** such as how the number of photos relates to adoption fees
    * **Machine learning predictions** of each pet’s probability of fast adoption
      
    Use these tools to spot pets that may need extra promotion or foster care, plan resources,
    and ultimately help more animals find homes.
    """
)

# ---------- Sidebar: upload ----------
st.sidebar.header("1) Upload data")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("2) Model info")
model_path = Path(__file__).with_name("model.pkl")  # /app/src/model.pkl

@st.cache_resource
def load_pipeline(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

pipe = load_pipeline(model_path)
if pipe is None:
    st.error("`model.pkl` not found next to the app (src/model.pkl).")
    st.stop()

st.sidebar.success("Model loaded.")

# ----- Sidebar: model evaluation -----
st.sidebar.markdown("---")
st.sidebar.header("Model Evaluation")
st.sidebar.write("Accuracy: 0.65")
st.sidebar.write("ROC AUC: 0.79")
st.sidebar.write("Precision (Fast adoption): 0.66")
st.sidebar.write("Recall (Fast adoption): 0.61")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Privacy note: This app processes uploaded CSVs in memory only. "
    "No files are stored server side."
)

# ---------- Helper: light cleaning that is safe even if columns are missing ----------
def light_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # soft caps / conversions that won't break if columns are missing
    if "Age" in out.columns:
        # Age is in months in Petfinder; cap extreme values to 240 (20 years)
        out["Age"] = pd.to_numeric(out["Age"], errors="coerce").clip(upper=240)
        out["AgeYears"] = (out["Age"] / 12).round(2)
    if "PhotoAmt" in out.columns:
        out["PhotoAmt"] = pd.to_numeric(out["PhotoAmt"], errors="coerce").fillna(0)
    if "Fee" in out.columns:
        out["Fee"] = pd.to_numeric(out["Fee"], errors="coerce").fillna(0)
    return out

# ---------- Load data ----------
source = "uploaded file" if uploaded else "sample_pets.csv"

try:
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    else:
        # Look for sample_pets.csv in the same folder as this script
        csv_path = Path(__file__).with_name("sample_pets.csv")
        df_raw = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not read {source}: {e}")
    st.stop()

# light cleanup + helper cols (e.g., AgeYears)
df = light_clean(df_raw)

st.subheader("Data Preview")
st.markdown(
    """
    Here’s a quick look at the first few rows of the default dataset (or the one you uploaded).
    This preview helps you double check that columns and values look right before the dashboard
    runs the visualizations and predictions.
    """
)
st.caption(f"Current Dataset: {source}")
st.dataframe(df.head(20), use_container_width=True)

# ---------- EDA ----------
st.subheader("Visual Insights")
st.markdown(
    "These visuals highlight important patterns in the data, such as "
    "typical adoption fees, the age distribution of pets, and how the "
    "number of photos relate to adoption fees."
)

eda_cols = st.columns(3)

with eda_cols[0]:
    # If AdoptionSpeed is present, show fast vs slow distribution
    if "Fee" in df.columns:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Fee:Q", bin=alt.Bin(maxbins=30), title="Adoption fee"),
                    y=alt.Y("count()", title="Count"),
                    tooltip=["count()"]
                )
                .properties(height=250)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption(
                "Shows how many pets fall into different adoption fee ranges. "
                "Helps identify common price points and outliers."
            )
    else:
        st.caption("Fee not found.")

with eda_cols[1]:
    if "AgeYears" in df.columns:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("AgeYears:Q", bin=alt.Bin(maxbins=30), title="Age (years)"),
                y=alt.Y("count()", title="Count"),
                tooltip=["count()"]
            )
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Shows the spread of pet ages in years. Younger animals often get adopted faster, "
            "so this gives a quick look at the population's age mix."
        )
    else:
        st.caption("Age column not found, skipping age histogram.")

with eda_cols[2]:
    # Photos vs Fee scatter (if available)
    if {"PhotoAmt", "Fee"} <= set(df.columns):
        chart = (
            alt.Chart(df)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X("PhotoAmt:Q", title="Number of photos"),
                y=alt.Y("Fee:Q", title="Adoption fee"),
                tooltip=list(df.columns)[:6]
            )
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            "Explores whether pets with more photos tend to have different adoption fees. "
            "Helps shelters decide how many photos to include in listings."
        )
    else:
        st.caption("PhotoAmt/Fee not found, skipping scatter.")

# ---------- Predictions using the Pipeline ----------
st.subheader("Predictions")
st.markdown(
    """
    Below are the predicted chances that each pet will be adopted quickly, based on the trained model.  
    * The **Adoption_Prob** column shows the probability of a fast adoption (closer to 1 means higher chance).  
    * The **Needs_Attention** column flags pets with a lower probability than the threshold you set in the slider
    so staff can focus extra marketing or foster efforts on those animals.
    """
)

# IMPORTANT: pass the **raw** frame to the pipeline (it contains original columns)
X_for_model = df.copy()

# Make sure we do not pass a target column if it exists
for t in ("AdoptionRate", "Adoption_Fast", "AdoptionFast", "Target"):
    if t in X_for_model.columns:
        X_for_model = X_for_model.drop(columns=[t])

try:
    proba = pipe.predict_proba(X_for_model)[:, 1]
except Exception as e:
    st.error("The uploaded CSV is missing one or more columns the pipeline expects.\n\n"
             "Tip: export a small CSV from your training set and use the same headers.")
    st.exception(e)
    st.stop()

df_pred = df.copy()
df_pred["Adoption_Prob"] = np.round(proba, 3)

# Attention threshold for decision support
thresh = st.slider(
    "Flag pets below this probability (need extra marketing promotion/fostering):",
    0.0, 1.0, 0.40, 0.01
)
df_pred["Needs_Attention"] = df_pred["Adoption_Prob"] < thresh

# Show a concise table
cols_to_show = ["Adoption_Prob", "Needs_Attention"] + [
    c for c in ["Name", "Age", "AgeYears", "PhotoAmt", "Fee", "Type", "Gender", "Health", "Breed1"]
    if c in df_pred.columns
]
st.dataframe(df_pred[cols_to_show].head(25), use_container_width=True)

# Download predictions
out_csv = df_pred.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full predictions as CSV",
    data=out_csv,
    file_name="adoption_predictions.csv",
    mime="text/csv"
)

# ---------- Extra visualization: probability distribution ----------
st.subheader("Adoption probability distribution")
st.markdown(
    """
    **What this shows:**  
    A histogram of the model’s predicted probabilities that each pet
    will be adopted quickly.  
    * Bars toward the **right (close to 1.0)** represent pets the model believes
      are very likely to be adopted soon.  
    * Bars toward the **left (close to 0.0)** represent pets predicted to need
      extra marketing or foster support.

    Shelters can use this chart to see the overall mix of “high-probability” versus
    “low-probability” pets in the current dataset, which helps with planning and prioritizing.
    """
)
prob_chart = (
    alt.Chart(pd.DataFrame({"prob": df_pred["Adoption_Prob"]}))
    .mark_bar()
    .encode(
        x=alt.X("prob:Q", bin=alt.Bin(maxbins=25), title="Predicted probability of fast adoption"),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"]
    )
).properties(height=280)
st.altair_chart(prob_chart, use_container_width=True)