import io
import traceback

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=np.float64).flatten(), np.asarray(b, dtype=np.float64).flatten()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 and norm_b == 0:
        return 100.0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), 0.0, 1.0) * 100)


def compare_text(f1, f2):
    from sklearn.feature_extraction.text import TfidfVectorizer

    t1 = f1.read().decode("utf-8", errors="replace")
    t2 = f2.read().decode("utf-8", errors="replace")

    if not t1.strip() or not t2.strip():
        return 0.0

    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit([t1, t2])
        v1 = vec.transform([t1]).toarray()[0]
        v2 = vec.transform([t2]).toarray()[0]
    except ValueError:
        return 0.0

    return cosine_similarity(v1, v2)


def compare_image(f1, f2):
    from PIL import Image

    size = (64, 64)
    img1 = Image.open(f1).convert("L").resize(size)
    img2 = Image.open(f2).convert("L").resize(size)

    return cosine_similarity(np.array(img1), np.array(img2))


def compare_audio(f1, f2):
    import librosa

    y1, sr1 = librosa.load(f1, sr=None)
    y2, sr2 = librosa.load(f2, sr=None)

    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)

    v1 = np.mean(mfcc1, axis=1)
    v2 = np.mean(mfcc2, axis=1)

    return cosine_similarity(v1, v2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error": "Both files are required"}), 400

    modality = request.form.get("modality", "").strip().lower()
    f1 = request.files["file1"]
    f2 = request.files["file2"]

    if f1.filename == "" or f2.filename == "":
        return jsonify({"error": "Both files must be selected"}), 400

    if modality not in ("text", "image", "audio"):
        return jsonify({"error": "Invalid modality. Choose text, image, or audio"}), 400

    try:
        if modality == "text":
            score = compare_text(f1, f2)
        elif modality == "image":
            score = compare_image(f1, f2)
        elif modality == "audio":
            score = compare_audio(f1, f2)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 422

    return jsonify({"similarity": round(score, 2)})


if __name__ == "__main__":
    app.run(debug=True)
