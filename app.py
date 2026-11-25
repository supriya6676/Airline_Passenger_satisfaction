"""
✈️ Airline Passenger Satisfaction Predictor
Streamlit app that serves the KNN model saved in Airline_model_pred.pkl

Pickle contents (dict):
    "model" -> KNeighborsClassifier (k=5, 27 features)
    "OHE"   -> OneHotEncoder for gender, customer_type, type_of_travel, class
    "LE"    -> LabelEncoder for target (Neutral or Dissatisfied / Satisfied)

Scaling: age, flight_distance, departure_delay and arrival_delay were
Min-Max scaled during training. The scaler itself was not saved in the
pickle, so the exact min/max values used in training are hard-coded below
(they were recovered from the model's training data).
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="SkyScore · Passenger Satisfaction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Airline_model_pred.pkl")

# Min-Max ranges used at training time  (x_scaled = (x - min) / (max - min))
MINMAX_RANGES = {
    "age": (7, 85),
    "flight_distance": (31, 4983),
    "departure_delay": (0, 1592),
    "arrival_delay": (0, 1584),
}

CAT_COLS = ["gender", "customer_type", "type_of_travel", "class"]

RATING_COLS = {
    "departure_and_arrival_time_convenience": "🕒 Departure / Arrival Time Convenience",
    "ease_of_online_booking": "💻 Ease of Online Booking",
    "check-in_service": "🧾 Check-in Service",
    "online_boarding": "📱 Online Boarding",
    "gate_location": "🚪 Gate Location",
    "on-board_service": "🤝 On-board Service",
    "seat_comfort": "💺 Seat Comfort",
    "leg_room_service": "🦵 Leg Room Service",
    "cleanliness": "🧼 Cleanliness",
    "food_and_drink": "🍽️ Food and Drink",
    "in-flight_service": "🛎️ In-flight Service",
    "in-flight_wifi_service": "📶 In-flight Wifi Service",
    "in-flight_entertainment": "🎬 In-flight Entertainment",
    "baggage_handling": "🧳 Baggage Handling",
}


RATING_ORDER = list(RATING_COLS)  # order of ratings in the presets below

# Real passengers from the dataset, used as one-click examples
SAMPLE_PASSENGERS = {
    "😊 Satisfied business traveller": dict(
        gender="Female", age=43, customer_type="Returning", type_of_travel="Business",
        travel_class="Business", flight_distance=580, departure_delay=0, arrival_delay=0,
        ratings=[3, 3, 3, 4, 3, 4, 5, 4, 3, 3, 4, 3, 4, 4],
    ),
    "😊 Satisfied long-haul flyer": dict(
        gender="Male", age=43, customer_type="Returning", type_of_travel="Business",
        travel_class="Business", flight_distance=2828, departure_delay=0, arrival_delay=0,
        ratings=[4, 4, 3, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4],
    ),
    "😞 Dissatisfied economy traveller": dict(
        gender="Male", age=18, customer_type="Returning", type_of_travel="Personal",
        travel_class="Economy", flight_distance=1199, departure_delay=5, arrival_delay=0,
        ratings=[3, 2, 2, 2, 1, 3, 3, 3, 3, 3, 5, 2, 3, 3],
    ),
}

DEFAULTS = dict(
    gender="Female", age=35, customer_type="Returning", type_of_travel="Business",
    travel_class="Business", flight_distance=1000, departure_delay=0, arrival_delay=0,
    ratings=[3] * len(RATING_COLS),
)


def load_profile(profile: dict):
    """Callback: push a profile's values into the form widgets."""
    for k, v in profile.items():
        if k == "ratings":
            for col, r in zip(RATING_ORDER, v):
                st.session_state[col] = r
        else:
            st.session_state[k] = v


# First run: fill the form with defaults
if "age" not in st.session_state:
    load_profile(DEFAULTS)

# ------------------------------------------------------------------
# CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: linear-gradient(160deg, #eef4ff 0%, #f8fbff 45%, #ffffff 100%);
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #0b1f4d 0%, #1446a0 55%, #2f80ed 100%);
        border-radius: 22px;
        padding: 2.2rem 2.5rem;
        color: #fff;
        box-shadow: 0 12px 30px rgba(20, 70, 160, 0.25);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "✈";
        position: absolute; right: 2rem; top: -0.5rem;
        font-size: 9rem; opacity: 0.12; transform: rotate(-20deg);
    }
    .hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; color: #fff; }
    .hero p  { margin: 0.4rem 0 0; font-size: 1.05rem; opacity: 0.9; }

    /* Section cards */
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.2rem 1.4rem 0.4rem;
        box-shadow: 0 6px 18px rgba(15, 40, 90, 0.08);
        border: 1px solid #e6edf8;
        margin-bottom: 1rem;
    }
    .card-title {
        font-weight: 600; font-size: 1.1rem; color: #0b1f4d;
        border-left: 5px solid #2f80ed; padding-left: 0.6rem; margin-bottom: 0.8rem;
    }

    /* Result boxes */
    .result {
        border-radius: 20px; padding: 1.8rem; text-align: center; color: #fff;
        box-shadow: 0 10px 26px rgba(0,0,0,0.15);
        animation: pop 0.45s ease-out;
    }
    .result h2 { margin: 0.2rem 0; font-size: 2rem; color: #fff; }
    .result p  { margin: 0; font-size: 1.05rem; opacity: 0.95; }
    .ok  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .bad { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    @keyframes pop { 0% {transform: scale(0.9); opacity: 0;} 100% {transform: scale(1); opacity: 1;} }

    /* Metric tiles */
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #e6edf8; border-radius: 14px;
        padding: 0.8rem 1rem; box-shadow: 0 4px 12px rgba(15, 40, 90, 0.06);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(120deg, #1446a0, #2f80ed);
        color: #fff; border: none; border-radius: 12px;
        padding: 0.7rem 1.4rem; font-weight: 600; font-size: 1.05rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px); color: #fff;
        box-shadow: 0 8px 18px rgba(47, 128, 237, 0.35);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: #e9f1ff; border-radius: 10px 10px 0 0;
        padding: 0.6rem 1.2rem; font-weight: 600; color: #1446a0;
    }
    .stTabs [aria-selected="true"] { background: #1446a0 !important; color: #fff !important; }

    .footer { text-align: center; color: #7a8bb0; font-size: 0.85rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts(path: str):
    bundle = joblib.load(path)
    return bundle["model"], bundle["OHE"], bundle["LE"]


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`. Put **Airline_model_pred.pkl** next to app.py.")
    st.stop()

model, ohe, le = load_artifacts(MODEL_PATH)


# ------------------------------------------------------------------
# Feature engineering + scaling (same steps as training)
# ------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'Flight Distance' -> 'flight_distance' (matches training column names)."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df = df.drop(columns=[c for c in ["id", "satisfaction"] if c in df.columns])

    # Missing arrival delay -> use departure delay (closest proxy), else 0
    if "arrival_delay" in df.columns:
        df["arrival_delay"] = df["arrival_delay"].fillna(df.get("departure_delay", 0)).fillna(0)

    # 1) One-hot encode categoricals with the saved encoder
    encoded = pd.DataFrame(
        ohe.transform(df[CAT_COLS]),
        columns=ohe.get_feature_names_out(),
        index=df.index,
    )
    df = pd.concat([encoded, df.drop(columns=CAT_COLS)], axis=1)

    # 2) Min-Max scale the numeric columns (clip to training range)
    for col, (lo, hi) in MINMAX_RANGES.items():
        df[col] = ((df[col].astype(float) - lo) / (hi - lo)).clip(0, 1)

    # 3) Same column order as the model
    return df[list(model.feature_names_in_)]


def predict(df: pd.DataFrame):
    X = preprocess(df)
    proba = model.predict_proba(X)
    labels = le.inverse_transform(model.predict(X))
    sat_idx = list(le.classes_).index("Satisfied")
    return labels, proba[:, sat_idx]


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>✈️ SkyScore — Passenger Satisfaction Predictor</h1>
        <p>Enter a passenger's journey details and service ratings to predict whether they'll be satisfied.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_about = st.tabs(["🎯 Single Prediction", "📂 Batch Prediction", "ℹ️ About the Model"])

# ---------------- Single prediction ----------------
with tab_single:
    st.markdown("**⚡ Quick fill — load an example passenger, then tweak any field:**")
    qcols = st.columns(len(SAMPLE_PASSENGERS) + 1)
    for qc, (name, prof) in zip(qcols, SAMPLE_PASSENGERS.items()):
        qc.button(name, on_click=load_profile, args=(prof,), use_container_width=True)
    qcols[-1].button("↺ Reset form", on_click=load_profile, args=(DEFAULTS,), use_container_width=True)
    st.write("")

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">👤 Passenger Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", list(ohe.categories_[0]), key="gender")
        age = c2.number_input("Age", min_value=7, max_value=85, step=1, key="age")
        customer_type = c1.selectbox("Customer Type", list(ohe.categories_[1]), key="customer_type")
        type_of_travel = c2.selectbox("Type of Travel", list(ohe.categories_[2]), key="type_of_travel")
        travel_class = st.selectbox("Class", list(ohe.categories_[3]), key="travel_class")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🛫 Flight Details</div>', unsafe_allow_html=True)
        flight_distance = st.number_input("Flight Distance (miles)", min_value=31, max_value=4983, step=10, key="flight_distance")
        c3, c4 = st.columns(2)
        departure_delay = c3.number_input("Departure Delay (min)", min_value=0, max_value=1592, step=1, key="departure_delay")
        arrival_delay = c4.number_input("Arrival Delay (min)", min_value=0, max_value=1584, step=1, key="arrival_delay")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="card"><div class="card-title">⭐ Service Ratings '
            '<span style="font-weight:400;font-size:0.85rem;color:#7a8bb0">(0 = not applicable, 1–5 = poor → excellent)</span></div>',
            unsafe_allow_html=True,
        )
        ratings = {}
        items = list(RATING_COLS.items())
        rc1, rc2 = st.columns(2)
        for i, (col, label) in enumerate(items):
            target = rc1 if i % 2 == 0 else rc2
            min_v = 1 if col == "baggage_handling" else 0
            ratings[col] = target.slider(label, min_value=min_v, max_value=5, key=col)
        st.markdown("</div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1, 1])
    go = mid.button("🚀 Predict Satisfaction", use_container_width=True)

    if go:
        row = {
            "gender": gender,
            "customer_type": customer_type,
            "type_of_travel": type_of_travel,
            "class": travel_class,
            "age": age,
            "flight_distance": flight_distance,
            "departure_delay": departure_delay,
            "arrival_delay": arrival_delay,
            **ratings,
        }
        labels, p_sat = predict(pd.DataFrame([row]))
        label, p = labels[0], float(p_sat[0])

        st.write("")
        r1, r2 = st.columns([1.3, 1], gap="large")
        with r1:
            if label == "Satisfied":
                st.markdown(
                    f'<div class="result ok"><p>Prediction</p><h2>😊 Satisfied</h2>'
                    f'<p>Confidence: {p*100:.0f}%</p></div>',
                    unsafe_allow_html=True,
                )
                st.balloons()
            else:
                st.markdown(
                    f'<div class="result bad"><p>Prediction</p><h2>😞 Neutral or Dissatisfied</h2>'
                    f'<p>Confidence: {(1-p)*100:.0f}%</p></div>',
                    unsafe_allow_html=True,
                )
        with r2:
            st.metric("Probability of Satisfaction", f"{p*100:.0f}%")
            st.progress(p)
            rated = [v for v in ratings.values() if v > 0]
            st.metric("Average Service Rating", f"{np.mean(rated):.2f} / 5" if rated else "—")

        st.markdown("#### 📊 Service Rating Profile")
        chart_df = pd.DataFrame(
            {"Rating": list(ratings.values())},
            index=[RATING_COLS[c].split(" ", 1)[1] for c in ratings],
        )
        st.bar_chart(chart_df, color="#2f80ed", horizontal=True)

        low = [RATING_COLS[c] for c, v in ratings.items() if 0 < v <= 2]
        if low:
            st.warning("**Areas pulling satisfaction down:** " + ", ".join(low))

# ---------------- Batch prediction ----------------
with tab_batch:
    st.markdown(
        '<div class="card"><div class="card-title">📂 Upload a CSV</div>'
        "<p>Use the same columns as the original dataset (e.g. <code>Gender</code>, <code>Age</code>, "
        "<code>Customer Type</code>, <code>Flight Distance</code>, service ratings…). "
        "<code>ID</code> and <code>Satisfaction</code> are optional.</p></div>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Choose CSV file", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        try:
            with st.spinner("Scoring passengers…"):
                labels, p_sat = predict(raw)
            out = raw.copy()
            out["Predicted Satisfaction"] = labels
            out["Satisfaction Probability"] = np.round(p_sat, 2)

            m1, m2, m3 = st.columns(3)
            m1.metric("Passengers scored", f"{len(out):,}")
            m2.metric("Predicted Satisfied", f"{(labels == 'Satisfied').mean()*100:.1f}%")
            if "Satisfaction" in raw.columns:
                acc = (raw["Satisfaction"].astype(str).str.strip().values == labels).mean()
                m3.metric("Accuracy vs. actual", f"{acc*100:.1f}%")
            else:
                m3.metric("Avg. satisfaction probability", f"{p_sat.mean()*100:.1f}%")

            st.dataframe(out.head(200), use_container_width=True)
            st.download_button(
                "⬇️ Download predictions",
                out.to_csv(index=False).encode("utf-8"),
                file_name="satisfaction_predictions.csv",
                mime="text/csv",
            )
        except KeyError as e:
            st.error(f"Missing column in uploaded file: {e}")
        except ValueError as e:
            st.error(f"Could not process file: {e}")

# ---------------- About ----------------
with tab_about:
    st.markdown(
        f"""
        <div class="card">
        <div class="card-title">🧠 Model</div>
        <ul>
          <li><b>Algorithm:</b> K-Nearest Neighbors (k = {model.n_neighbors}, {model.metric} distance, {model.weights} weights)</li>
          <li><b>Input features:</b> {model.n_features_in_}</li>
          <li><b>Classes:</b> {", ".join(le.classes_)}</li>
        </ul>
        </div>
        <div class="card">
        <div class="card-title">⚙️ Preprocessing Pipeline</div>
        <ol>
          <li><b>Column cleaning:</b> names lower-cased, spaces → underscores; <code>ID</code> dropped.</li>
          <li><b>One-Hot Encoding</b> (saved <code>OHE</code>): Gender, Customer Type, Type of Travel, Class.</li>
          <li><b>Min-Max Scaling</b> to 0–1: Age (7–85), Flight Distance (31–4983),
              Departure Delay (0–1592), Arrival Delay (0–1584).</li>
          <li><b>Service ratings</b> (0–5) used as-is.</li>
          <li><b>Label Encoding</b> (saved <code>LE</code>) for the target.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">Built with Streamlit · Airline Passenger Satisfaction ML Project</div>', unsafe_allow_html=True)
