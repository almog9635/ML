import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["USE_TF"] = "0"
import tensorflow as tf
import keras
model = keras.saving.load_model("model/truck_delivery_model.keras",
                                compile=False, safe_mode=False)
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re, joblib
from collections import Counter
import json
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')

idf_values = joblib.load("model/idf_values.pkl")
training_columns = joblib.load("model/training_columns.pkl")
mlb = joblib.load("model/mlb.pkl")

model = tf.keras.models.load_model("model/truck_delivery_model.keras")

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.discard('not')

kmeans = joblib.load("model/kmeans_3clusters.joblib")        # ➌ NEW
with open("model/cluster_names.json") as f:                  # ➍ NEW
    CLUSTER_NAMES = {int(k): v for k, v in json.load(f).items()}
encoder = SentenceTransformer(                               # ➎ NEW
    'all-MiniLM-L6-v2', cache_folder="model/transformer_cache"
)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return words


def compute_tf(word_dict, doc):
    total_words = len(doc)
    return {word: count / total_words for word, count in word_dict.items()} if total_words else {}


def compute_tfidf(tf_dict, idf_dict):
    return {word: tf_dict[word] * idf_dict[word] for word in tf_dict if word in idf_dict}


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    texts = data.get('texts', [])
    print(texts)

    processed_texts = [preprocess_text(text) for text in texts]
    word_counts = [Counter(doc) for doc in processed_texts]
    tf_values = [compute_tf(counter, doc) for counter, doc in zip(word_counts, processed_texts)]
    tfidf_values = [compute_tfidf(tf_dict, idf_values) for tf_dict in tf_values]

    X_new_df = pd.DataFrame(tfidf_values).fillna(0)
    X_new_df = X_new_df.reindex(columns=training_columns, fill_value=0)

    predictions = model.predict(X_new_df)
    threshold = 0.5
    y_pred_binary = (predictions > threshold).astype(int)
    predicted_tags = mlb.inverse_transform(y_pred_binary)

    response = [{"text": text, "predicted_tags": tags} for text, tags in zip(texts, predicted_tags)]
    print(response)
    return jsonify(response)

@app.route('/cluster', methods=['POST'])
def cluster():
    data = request.json
    sentences = data.get('sentences', [])
    if not isinstance(sentences, list) or not sentences:
        return jsonify({"error": '"sentences" must be a non-empty list'}), 400

    embeddings = encoder.encode(sentences)
    cluster_ids = kmeans.predict(embeddings).tolist()
    names = [CLUSTER_NAMES.get(i, f"Cluster {i}") for i in cluster_ids]

    return jsonify({"cluster_id": cluster_ids, "cluster_name": names})


if __name__ == '__main__':
    app.run(debug=True, port=4005)
