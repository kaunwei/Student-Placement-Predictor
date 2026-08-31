
import streamlit as st
import pickle
import numpy as np

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: #f7f9fc;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #667085;
        margin-bottom: 35px;
    }

    /* Input card */
    .input-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        margin-bottom: 25px;
    }

    /* Result card */
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        margin-top: 25px;
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .result-text {
        font-size: 17px;
        color: #667085;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98A2B3;
        font-size: 13px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 Student Placement Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter your academic information to estimate your placement outcome.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.markdown('<div class="input-card">', unsafe_allow_html=True)

st.subheader("Student Information")

cgpa = st.number_input(
    "📚 CGPA",
    min_value=0.0,
    max_value=4.0,
    value=4.0,
    step=0.1,
    help="Enter your CGPA on a scale of 0 to 4."
)

iq = st.number_input(
    "🧠 IQ Score",
    min_value=50,
    max_value=250,
    value=120,
    step=1,
    help="Enter your IQ score."
)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("🔍 Predict Placement"):

    # Model expects:
    # [CGPA, IQ]

    input_data = np.array([[cgpa, iq]])

    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0]

    placement_probability = probability[1] * 100

    # ------------------------------------------------
    # PLACED
    # ------------------------------------------------

    if prediction == 1:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-title">
                    🎉 Placement Predicted
                </div>

                <div class="result-text">
                    Based on the provided CGPA and IQ score,
                    the model predicts that the student is likely
                    to be placed.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Placement Probability",
            f"{placement_probability:.2f}%"
        )

        st.progress(int(placement_probability))

    # ------------------------------------------------
    # NOT PLACED
    # ------------------------------------------------

    else:

        st.markdown(
            """
            <div class="result-card">
                <div class="result-title">
                    📈 Placement Not Predicted
                </div>

                <div class="result-text">
                    Based on the provided CGPA and IQ score,
                    the model predicts that the student may
                    not be placed.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Placement Probability",
            f"{placement_probability:.2f}%"
        )

        st.progress(int(placement_probability))


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Student Placement Prediction System • Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

