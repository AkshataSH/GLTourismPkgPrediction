
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


DATASET_REPO_ID = "akshatash/GLTourismPkgPrediction"
MODEL_REPO_ID = "akshatash/GLTourismPkgPrediction"
MODEL_FILE = "best_tourism_model_v1.joblib"
REFERENCE_DATA_FILE = "tourism.csv"

FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

NUMERICAL_COLS_TO_STANDARDIZE = [
    "Age",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "DurationOfPitch",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]


st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILE,
        repo_type="model",
    )
    return joblib.load(model_path)


@st.cache_data
def load_reference_statistics():
    data_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=REFERENCE_DATA_FILE,
        repo_type="dataset",
    )
    reference_df = pd.read_csv(data_path)
    reference_df = reference_df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")
    reference_df["Gender"] = (
        reference_df["Gender"]
        .astype(str)
        .str.strip()
        .replace({"Fe Male": "Female", "Fe male": "Female"})
    )

    fill_values = {
        "Age": reference_df["Age"].mean(),
        "MonthlyIncome": reference_df["MonthlyIncome"].mean(),
    }

    for col, value in fill_values.items():
        reference_df[col] = reference_df[col].fillna(value)

    standardization_values = {}
    for col in NUMERICAL_COLS_TO_STANDARDIZE:
        standardization_values[col] = {
            "mean": reference_df[col].mean(),
            "std": reference_df[col].std(),
        }

    return fill_values, standardization_values


def prepare_input_data(input_df, fill_values, standardization_values):
    df = input_df.copy()

    df = df.drop(columns=["Unnamed: 0", "CustomerID", "ProdTaken"], errors="ignore")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[FEATURE_COLUMNS]

    numeric_cols = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Gender"] = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .replace({"Fe Male": "Female", "Fe male": "Female"})
    )
    df["TypeofContact"] = df["TypeofContact"].astype(str).str.strip()
    df["Occupation"] = df["Occupation"].astype(str).str.strip()
    df["MaritalStatus"] = df["MaritalStatus"].astype(str).str.strip()
    df["ProductPitched"] = df["ProductPitched"].astype(str).str.strip()
    df["Designation"] = df["Designation"].astype(str).str.strip()

    df["Age"] = df["Age"].fillna(fill_values["Age"])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(fill_values["MonthlyIncome"])

    df["TypeofContact"] = df["TypeofContact"].map(
        {
            "Company Invited": 0,
            "Self Inquiry": 1,
            "Self Enquiry": 1,
        }
    )
    df["CityTier"] = df["CityTier"] - 1
    df["Occupation"] = df["Occupation"].map(
        {
            "Salaried": 0,
            "Freelancer": 1,
            "Free Lancer": 1,
            "Small Business": 2,
            "Large Business": 3,
        }
    )
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["MaritalStatus"] = df["MaritalStatus"].map(
        {
            "Single": 0,
            "Married": 1,
            "Divorced": 2,
            "Unmarried": 3,
        }
    )

    for col in NUMERICAL_COLS_TO_STANDARDIZE:
        mean_value = standardization_values[col]["mean"]
        std_value = standardization_values[col]["std"]
        df[col] = (df[col] - mean_value) / std_value

    return df


def add_predictions(input_df, prepared_df, model):
    predictions = model.predict(prepared_df)
    result_df = input_df.copy()
    result_df["Prediction"] = predictions
    result_df["Prediction_Label"] = result_df["Prediction"].map(
        {
            0: "No Purchase",
            1: "Purchase",
        }
    )

    if hasattr(model, "predict_proba"):
        result_df["Purchase_Probability"] = model.predict_proba(prepared_df)[:, 1]

    return result_df


model = load_model()
fill_values, standardization_values = load_reference_statistics()

st.title("Tourism Package Purchase Predictor")
st.caption("Predict whether a customer is likely to buy a tourism package.")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Model", "XGBoost")
metric_col2.metric("Input Modes", "Single")
metric_col3.metric("Target", "ProdTaken")

tabs = st.tabs(["Single Prediction"])
tab_single = tabs[0]

with tab_single:
    st.subheader("Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        citytier = st.selectbox("City Tier", [1, 2, 3], index=2)
        durationofpitch = st.number_input("Duration of Pitch", min_value=0.0, max_value=150.0, value=15.0, step=1.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        numberofpersonvisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=10, value=3, step=1)
        numberoffollowups = st.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0, step=1.0)
        productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        numberoftrips = st.number_input("Number of Trips", min_value=0.0, max_value=30.0, value=2.0, step=1.0)

    with col3:
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2)
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=300000.0, value=25000.0, step=1000.0)

    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": typeofcontact,
                "CityTier": citytier,
                "DurationOfPitch": durationofpitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": numberofpersonvisiting,
                "NumberOfFollowups": numberoffollowups,
                "ProductPitched": productpitched,
                "PreferredPropertyStar": preferredpropertystar,
                "MaritalStatus": maritalstatus,
                "NumberOfTrips": numberoftrips,
                "Passport": passport,
                "PitchSatisfactionScore": pitchsatisfactionscore,
                "OwnCar": owncar,
                "NumberOfChildrenVisiting": numberofchildrenvisiting,
                "Designation": designation,
                "MonthlyIncome": monthlyincome,
            }
        ]
    )

    if st.button("Predict Customer", type="primary"):
        prepared_data = prepare_input_data(input_data, fill_values, standardization_values)
        result = add_predictions(input_data, prepared_data, model)

        prediction_label = result.loc[0, "Prediction_Label"]
        probability = result.loc[0, "Purchase_Probability"] if "Purchase_Probability" in result.columns else None

        if prediction_label == "Purchase":
            st.success("Prediction: Customer is likely to purchase the tourism package.")
        else:
            st.warning("Prediction: Customer is not likely to purchase the tourism package.")

        if probability is not None:
            st.metric("Purchase Probability", f"{probability:.2%}")

        with st.expander("Prepared model input"):
            st.dataframe(prepared_data, use_container_width=True)


