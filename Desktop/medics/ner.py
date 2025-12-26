"""
This file is responsible for identifying medical-related terms from user input.

It uses the SpaCy language model to process the text and then matches
words against a predefined list of medical keywords such as diseases,
symptoms, and treatments.

it will going to return the detected medical word, the category it belongs to (Disease, Symptom, or Treatment)

This helps the chatbot understand important medical concepts
mentioned in the question.
"""

import spacy

nlp = spacy.load("en_core_web_sm")

MEDICAL_KEYWORDS = {
    "DISEASE": ["diabetes", "cancer", "asthma", "infection", "hypertension"],
    "SYMPTOM": ["pain", "fever", "cough", "fatigue", "headache"],
    "TREATMENT": ["surgery", "therapy", "medication", "chemotherapy"]
}

def extract_medical_entities(text):
    doc = nlp(text.lower())
    entities = []

    for token in doc:
        for label, keywords in MEDICAL_KEYWORDS.items():
            if token.text in keywords:
                entities.append((token.text, label))

    return entities
