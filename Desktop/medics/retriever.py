"""
This code contains the logic for searching and retrieving medical answers.
It uses a Sentence Transformer model to convert medical questions into numerical vectors. 
These embeddings help the systemunderstand the meaning of questions, not just exact words.

How it works:
1. Loads medical question-answer data from a JSON file.
2. Converts all questions into embeddings using a pre-trained model.
3. When a user asks a question, it converts the query into an embedding.
4. Finds the most similar question from the dataset.
5. Returns the corresponding medical answer.

This retriever allows the chatbot to give relevant answers
even when the question wording is different.
"""

import json
from sentence_transformers import SentenceTransformer, util

class MedicalRetriever:
    def __init__(self, data_path="data/medquad.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.questions = [item["question"] for item in self.data]
        self.answers = [item["answer"] for item in self.data]

        self.embeddings = self.model.encode(
            self.questions,
            convert_to_tensor=True
        )

    def retrieve(self, query, top_k=1):
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        top_results = scores.topk(top_k)

        results = []
        for idx in top_results.indices:
            results.append(self.answers[idx])

        return results
