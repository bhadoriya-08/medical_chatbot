"""this code is used to prepare the MedQuAD dataset for the Medical Q&A Chatbot.

It goes through all XML files inside the MedQuAD dataset folder,
extracts medical question-answer pairs, and converts them into
a single JSON file.

The final output file (medquad.json) is later used for searching
and retrieving medical answers.
"""

import os
import json
import xml.etree.ElementTree as ET

DATA_DIR = "MedQuAD-master"
output = []
failed_files = 0

for root_dir, _, files in os.walk(DATA_DIR):
    for file in files:
        if not file.lower().endswith(".xml"):
            continue

        file_path = os.path.join(root_dir, file)

        try:
           
            with open(file_path, "rb") as f:
                tree = ET.parse(f)

            root = tree.getroot()

           
            for qa in root.iter():
                tag = qa.tag.lower()
                if tag.endswith("qapair"):
                    q = qa.find(".//Question")
                    a = qa.find(".//Answer")

                    if q is not None and a is not None:
                        if q.text and a.text:
                            output.append({
                                "question": q.text.strip(),
                                "answer": a.text.strip()
                            })

        except Exception as e:
            failed_files += 1
            print(f"Skipped file: {file_path}")

print("\n==============================")
print("Total Q&A pairs extracted:", len(output))
print("Failed XML files:", failed_files)
print("==============================")

os.makedirs("data", exist_ok=True)

with open("data/medquad.json", "w", encoding="utf-8") as f: '''traversing the file'''
    json.dump(output, f, indent=2, ensure_ascii=False)
