import pandas as pd
import numpy as np
import pickle


# =========================
# LOAD FILES
# =========================

with open('scaler (2).pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('target_encoders (1).pkl', 'rb') as f:
    target_encoder = pickle.load(f)

with open('label_encoders (1).pkl', 'rb') as f:
    label_encoders = pickle.load(f)


# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_input(df):

    # =========================
    # ORDINAL ENCODING
    # =========================

    month_map = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }

    lead_time_map = {
        'short_term': 1,
        'medium_term': 2,
        'long_term': 3,
        'last_minute': 4
    }

    season_map = {
        'normal': 1,
        'peak': 2,
        'holiday': 3
    }

    df['arrival_date_month'] = df['arrival_date_month'].map(month_map)
    df['lead_time_group'] = df['lead_time_group'].map(lead_time_map)
    df['season'] = df['season'].map(season_map)

    # =========================
    # TARGET ENCODING
    # =========================

    target_cols = [
        'country',
        'market_segment',
        'distribution_channel',
        'reserved_room_type'
    ]

    # target encoding per kolom
    for col in target_cols:
        df[f"{col}_target"] = target_encoder[col].transform(df[[col]])

    # drop kolom asli
    df.drop(columns=target_cols, inplace=True)

    # =========================
    # LABEL ENCODING
    # =========================

    label_cols = [
        'hotel',
        'meal',
        'deposit_type',
        'customer_type',
        'guest_type',
        'is_loyal'
    ]

    for col in label_cols:
        le = label_encoders[col]

        # handle unseen labels
        df[col] = df[col].apply(
        lambda x: x if x in le.classes_ else le.classes_[0])
        
        df[col] = le.transform(df[col])

    # =========================
    # LOG TRANSFORM
    # =========================

    log_cols = [
        'lead_time',
        'adr',
        'Revenue',
        'total_stay',
        'days_in_waiting_list',
        'price_per_person'
    ]

    for col in log_cols:
        df[f'log_{col}'] = np.log(df[col] + 1)

    df.drop(columns=log_cols, inplace=True)

    # =========================
    # REORDER COLUMNS
    # =========================
    # Tambahkan kolom yang belum ada
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # urutkan sesuai training
    df = df[feature_names]

    # =========================
    # SCALING
    # =========================

    df = scaler.transform(df)

    return df