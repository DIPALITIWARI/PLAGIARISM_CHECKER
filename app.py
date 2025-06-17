from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import logging
import docx2txt
import PyPDF2

app = Flask(__name__)  
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
HOLD = 80
model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("train.csv")
df["processed"] = df["description_y"].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)

def get_embeddings_by_subject(subject):
    subject_map = {
        "daa": "DAA",
        "oops": "OOPs",
        "ml": "Machine Learning Algorithms",
        "cyber": "Cyber Security Theory"
    }
    selected = subject_map.get(subject.lower(), "")
    filtered_df = df[df['subject'].str.lower() == selected.lower()]
    texts = filtered_df["processed"].tolist()
    embeddings = model.encode(texts, convert_to_tensor=False)
    return texts, embeddings

def find_matches(text, subject, top_n=3):
    text = text.lower().strip()
    user_embedding = model.encode([text])
    corpus_texts, corpus_embeddings = get_embeddings_by_subject(subject)
    if not corpus_texts:
        return []
    scores = cosine_similarity(user_embedding, corpus_embeddings)[0]
    top_indices = np.argsort(scores)[-top_n:][::-1]
    return [{"text": corpus_texts[i], "score": round(scores[i] * 100, 2)} for i in top_indices]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/check_plagiarism", methods=["POST"])
def check_plagiarism():
    try:
        data = request.get_json()
        user_text = data.get("user_text", "").strip()
        subject = data.get("topic", "").strip()

        if not user_text or not subject:
            return jsonify({"error": "Text or subject missing"}), 400

        matches = find_matches(user_text, subject)
        if not matches:
            return jsonify({"similarity_score": 0, "matches": [], "plagiarism_detected": False})
        best_score = matches[0]["score"]
        return jsonify({
            "similarity_score": best_score,
            "matches": matches,
            "plagiarism_detected": bool(best_score >= PLAGIARISM_THRESHOLD)
        })
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/upload_document", methods=["POST"])
def upload_document():
    try:
        file = request.files.get("file")
        subject = request.form.get("topic", "").strip()

        if not file or not subject:
            return jsonify({"error": "File or subject missing"}), 400

        ext = file.filename.lower()
        if ext.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif ext.endswith(".docx"):
            text = docx2txt.process(file)
        elif ext.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        matches = find_matches(text, subject)
        if not matches:
            return jsonify({"similarity_score": 0, "matches": [], "plagiarism_detected": False})
        best_score = matches[0]["score"]
        return jsonify({
            "similarity_score": best_score,
            "matches": matches,
            "plagiarism_detected": bool(best_score >= PLAGIARISM_THRESHOLD)
        })

    except Exception as e:
        return jsonify({"error": f"File processing error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
