import streamlit as st
import pandas as pd
import pickle

from preprocessing import preprocess_input


# =========================
# LOAD MODEL
# =========================
with open('xgb_best (3).pkl', 'rb') as f:
    model, feature_names = pickle.load(f)


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="centered"
)


# =========================
# TITLE
# =========================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    """
    Predict whether a hotel booking
    will be canceled or not.
    """
)


# =========================
# INPUTS
# =========================

hotel = st.selectbox(
    'Hotel Type',
    ['City Hotel', 'Resort Hotel']
)

lead_time = st.number_input(
    'Lead Time',
    min_value=0,
    value=30
)

arrival_date_month = st.selectbox(
    'Arrival Month',
    [
        'January', 'February', 'March',
        'April', 'May', 'June',
        'July', 'August', 'September',
        'October', 'November', 'December'
    ]
)

lead_time_group = st.selectbox(
    'Lead Time Group',
    [
        'short_term',
        'medium_term',
        'long_term',
        'last_minute'
    ]
)

country = st.selectbox(
    'Country',
    ['PRT', 'GBR', 'FRA', 'ESP', 'DEU']
)

market_segment = st.selectbox(
    'Market Segment',
    [
        'Online TA',
        'Offline TA/TO',
        'Direct',
        'Groups',
        'Corporate'
    ]
)

distribution_channel = st.selectbox(
    'Distribution Channel',
    [
        'TA/TO',
        'Direct',
        'Corporate'
    ]
)

reserved_room_type = st.selectbox(
    'Reserved Room Type',
    ['A', 'B', 'C', 'D']
)

meal = st.selectbox(
    'Meal Type',
    ['BB', 'HB', 'FB', 'SC']
)

deposit_type = st.selectbox(
    'Deposit Type',
    [
        'No Deposit',
        'Refundable',
        'Non Refund'
    ]
)

customer_type = st.selectbox(
    'Customer Type',
    [
        'Transient',
        'Contract',
        'Transient-Party',
        'Group'
    ]
)

guest_type = st.selectbox(
    'Guest Type',
    [
        'Single',
        'Couple',
        'Family'
    ]
)

season = st.selectbox(
    'Season',
    [
        'normal',
        'peak',
        'holiday'
    ]
)

adr = st.number_input(
    'ADR',
    min_value=0.0,
    value=100.0
)

total_stay = st.number_input(
    'Total Stay',
    min_value=1,
    value=2
)

days_in_waiting_list = st.number_input(
    'Days in Waiting List',
    min_value=0,
    value=0
)

Revenue = st.number_input(
    'Revenue',
    min_value=0.0,
    value=200.0
)

price_per_person = st.number_input(
    'Price Per Person',
    min_value=0.0,
    value=50.0
)

is_loyal = st.selectbox(
    'Loyal Customer',
    ['yes', 'no']
)


# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict Cancellation"):

    input_data = pd.DataFrame({

        'hotel': [hotel],
        'lead_time': [lead_time],
        'arrival_date_month': [arrival_date_month],
        'lead_time_group': [lead_time_group],
        'country': [country],
        'market_segment': [market_segment],
        'distribution_channel': [distribution_channel],
        'reserved_room_type': [reserved_room_type],
        'meal': [meal],
        'deposit_type': [deposit_type],
        'customer_type': [customer_type],
        'guest_type': [guest_type],
        'season': [season],
        'adr': [adr],
        'Revenue': [Revenue],
        'total_stay': [total_stay],
        'days_in_waiting_list': [days_in_waiting_list],
        'price_per_person': [price_per_person],
        'is_loyal': [is_loyal]
    })

    processed_data = preprocess_input(input_data)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Booking will likely be canceled ❌\n\nProbability: {probability:.2%}"
        )

    else:
        st.success(
            f"Booking will likely NOT be canceled ✅\n\nProbability: {probability:.2%}"
        )