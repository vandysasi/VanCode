#!/usr/bin/env python3
"""
Streamlit App for Wellness Tourism Package Prediction.

Loads the model committed to this same folder (best_model.joblib) by the
CI pipeline, collects customer details through a form, and predicts
purchase likelihood. Because the model pipeline one-hot encodes categorical
columns internally, this app can submit raw human-readable values directly
-- there is no manual label-encoding map to keep in sync with training.
"""
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.joblib")


@st.cache_resource
def load_model():
    """Load the trained model committed to the repository."""
    return joblib.load(MODEL_PATH)


def main():
    st.title("Tourism Package Prediction")
    st.markdown("### Predict Customer Purchase Likelihood for the Wellness Tourism Package")
    st.markdown("---")

    model = load_model()

    st.sidebar.header("Customer Information")

    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 18, 80, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    st.sidebar.subheader("Location & Contact")
    city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    type_of_contact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

    st.sidebar.subheader("Professional Info")
    occupation = st.sidebar.selectbox(
        "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    designation = st.sidebar.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.sidebar.number_input("Monthly Income", 1000, 200000, 20000)

    st.sidebar.subheader("Travel Preferences")
    num_person_visiting = st.sidebar.slider("Number of Persons Visiting", 1, 5, 2)
    num_children_visiting = st.sidebar.slider("Number of Children Visiting", 0, 3, 0)
    preferred_property_star = st.sidebar.select_slider(
        "Preferred Property Star Rating", options=[3.0, 4.0, 5.0], value=3.0
    )
    num_trips = st.sidebar.slider("Number of Trips per Year", 0, 10, 2)

    st.sidebar.subheader("Additional Info")
    passport = st.sidebar.selectbox("Has Passport", ["Yes", "No"])
    own_car = st.sidebar.selectbox("Owns Car", ["Yes", "No"])

    st.sidebar.subheader("Sales Interaction")
    duration_of_pitch = st.sidebar.slider("Duration of Pitch (minutes)", 1, 60, 15)
    product_pitched = st.sidebar.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    num_followups = st.sidebar.slider("Number of Followups", 0, 6, 3)
    pitch_satisfaction_score = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Customer Profile Summary")
        profile_data = {
            "Age": age, "Gender": gender, "Marital Status": marital_status,
            "City Tier": city_tier, "Occupation": occupation,
            "Monthly Income": f"₹{monthly_income:,}",
            "Number of Persons": num_person_visiting,
            "Preferred Star Rating": preferred_property_star,
            "Annual Trips": num_trips, "Has Passport": passport, "Owns Car": own_car,
        }
        for key, value in profile_data.items():
            st.write(f"**{key}:** {value}")

    with col2:
        st.subheader("Prediction")

        if st.button("Predict Purchase Likelihood", type="primary"):
            input_df = pd.DataFrame([{
                "Age": age,
                "TypeofContact": type_of_contact,
                "CityTier": city_tier,
                "DurationOfPitch": duration_of_pitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": num_person_visiting,
                "NumberOfFollowups": num_followups,
                "ProductPitched": product_pitched,
                "PreferredPropertyStar": preferred_property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": num_trips,
                "Passport": 1 if passport == "Yes" else 0,
                "PitchSatisfactionScore": pitch_satisfaction_score,
                "OwnCar": 1 if own_car == "Yes" else 0,
                "NumberOfChildrenVisiting": num_children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
            }])

            try:
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]

                if prediction == 1:
                    st.success("High likelihood of purchase!")
                    st.write(f"**Confidence:** {prediction_proba[1]:.2%}")
                    st.balloons()
                else:
                    st.warning("Low likelihood of purchase")
                    st.write(f"**Confidence:** {prediction_proba[0]:.2%}")

                st.subheader("Probability Breakdown")
                prob_df = pd.DataFrame({
                    "Outcome": ["Will Not Purchase", "Will Purchase"],
                    "Probability": [prediction_proba[0], prediction_proba[1]],
                })
                st.bar_chart(prob_df.set_index("Outcome"))

            except Exception as e:
                st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.markdown("### About This Model")
    st.info(
        "This ML model predicts customer purchase likelihood for the Wellness "
        "Tourism Package based on demographics, travel preferences, and sales "
        "interaction data."
    )


if __name__ == "__main__":
    main()
