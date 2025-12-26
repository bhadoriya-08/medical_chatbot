"""
This is the main application code for the Medical Q&A Chatbot.

It uses Streamlit to create a simple web interface where users can:
1. Enter a medical-related question.
2. Search answers from the MedQuAD medical dataset using a retriever.
3. Display the most relevant medical answer.
4. Detect and highlight medical entities or diseases from the question
   using a Named Entity Recognition (NER) model.

The chatbot is designed for educational purposes and helps users
understand medical information in a question-and-answer format.
"""
import streamlit as st
from retriever import MedicalRetriever
from ner import extract_medical_entities

st.set_page_config(page_title="Medical Q&A Chatbot", layout="centered")

st.title("Medical Q&A Chatbot")
st.markdown("Ask medical questions based on the **MedQuAD dataset**.")

@st.cache_resource
def load_retriever():
    return MedicalRetriever()

retriever = load_retriever()

user_question = st.text_input("Enter your medical question:")

if st.button("Get Answer") and user_question:
    with st.spinner("Searching medical knowledge..."):
        answers = retriever.retrieve(user_question)
        entities = extract_medical_entities(user_question)

    st.subheader(" Answer")
    st.write(answers[0])

    if entities:
        st.subheader(" Detected Medical Entities or Disease")
        for ent, label in entities:
            st.write(f"- **{ent}** ({label})")

st.markdown("---")
st.caption("This chatbot is for educational purposes only.")

