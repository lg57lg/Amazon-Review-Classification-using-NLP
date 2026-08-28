import streamlit as st
import joblib
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Review Sentiment Classifier",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD MODEL & VECTORIZER
# -----------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")
    return vectorizer, model

vectorizer, model = load_artifacts()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💬 Sentiment Classifier")
st.sidebar.markdown(
    """
    This app predicts whether a product review expresses a
    **Positive** or **Negative** sentiment.

    **Model:** Multinomial Naive Bayes
    **Features:** TF-IDF (bag of words)
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Example reviews to try:**")
examples = [
    "Amazing product, excellent quality and fast delivery!",
    "Completely disappointed, the item stopped working and battery drains fast.",
    "Packaging was damaged and the phone screen is fake.",
    "Highly recommended, works perfectly and worth every penny.",
]
for ex in examples:
    if st.sidebar.button(ex, key=ex, use_container_width=True):
        st.session_state["review_text"] = ex

# -----------------------------
# MAIN
# -----------------------------
st.title("💬 Product Review Sentiment Classifier")
st.write("Enter a product review below and the model will predict its sentiment.")

if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

review_text = st.text_area(
    "Review text",
    value=st.session_state["review_text"],
    height=150,
    placeholder="e.g. The product quality is excellent and delivery was fast..."
)

predict_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

if predict_clicked:
    if not review_text.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        # Transform input text using the fitted TF-IDF vectorizer
        features = vectorizer.transform([review_text])

        # Predict class and probabilities
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, proba))
        confidence = prob_dict[prediction] * 100

        st.markdown("---")

        if prediction == "Positive":
            st.success(f"### ✅ Predicted Sentiment: {prediction}")
        else:
            st.error(f"### ❌ Predicted Sentiment: {prediction}")

        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(min(int(confidence), 100))

        st.markdown("#### Class Probabilities")
        for cls in classes:
            st.write(f"**{cls}**: {prob_dict[cls] * 100:.1f}%")
            st.progress(float(prob_dict[cls]))

        # Show which vocabulary words from the model were found in the input
        found_terms = [
            term for term in vectorizer.get_feature_names_out()
            if term in review_text.lower()
        ]
        if found_terms:
            st.markdown("#### Recognized Keywords")
            st.write(", ".join(sorted(found_terms)))
        else:
            st.info(
                "No known vocabulary words were found in this review — "
                "the model's vocabulary is limited, so predictions on "
                "unfamiliar wording may be unreliable."
            )

st.markdown("---")
with st.expander("ℹ️ About this model"):
    st.write(
        f"""
        - **Vectorizer:** TF-IDF over a fixed vocabulary of
          {len(vectorizer.vocabulary_)} words extracted from product-review
          training data.
        - **Classifier:** Multinomial Naive Bayes, trained to distinguish
          between **Positive** and **Negative** review sentiment.
        - Because the vocabulary is small and domain-specific (product
          review language: quality, delivery, battery, packaging, etc.),
          the model works best on similarly worded reviews and may be less
          reliable on text far outside that domain.
        """
    )
