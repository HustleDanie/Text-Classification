import streamlit as st
from transformers import pipeline

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Multi-Task NLP Classifier", layout="centered")

@st.cache_resource
def load_models():
    return {
        "spam": pipeline("text-classification", model="mariagrandury/roberta-base-finetuned-sms-spam-detection"),
        "topic": pipeline("text-classification", model="textattack/distilbert-base-uncased-ag-news"),
        "intent": pipeline("text-classification", model="Falconsai/intent_classification"),
        "language": pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection"),
        "news": pipeline("text-classification", model="mrm8488/distilroberta-finetuned-age_news-classification")
    }

models = load_models()

st.title("🧠 Multi-Task Text Classification App")
st.markdown("Classify text into **Spam**, **Topic**, **Intent**, **Language**, or **News Category** using Hugging Face Transformers.")

task = st.selectbox("Select a task", list(models.keys()))
text = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Classify"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            result = models[task](text)
            label = result[0]['label']
            score = result[0]['score']
            st.success(f"**Prediction:** {label}")
            st.metric("Confidence", f"{score:.4f}")
