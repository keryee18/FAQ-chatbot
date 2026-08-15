"""Streamlit UI for the TAR UMT FAQ chatbot.

Run with: streamlit run app.py
"""
import json
import random 
import numpy as np
import pandas as pd
from pathlib import Path

import streamlit as st
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DATA_FILE = Path(__file__).with_name("intents.json")
FALLBACK = "I’m not confident I understand that question. Please try rephrasing it or ask about programmes, admissions, fees, the library, intakes, scholarships or campus location."


@st.cache_data
def load_intents():
    with DATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)["intents"]


def make_examples(intents):
    texts, labels = [], []
    for intent in intents:
        texts.extend(intent["patterns"])
        labels.extend([intent["tag"]] * len(intent["patterns"]))
    return texts, labels


def residual_block(inputs, units, dropout_rate, name):
    """A two-layer residual block for numeric text features."""
    shortcut = inputs
    x = tf.keras.layers.Dense(units, activation="relu", name=f"{name}_dense_1")(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Dense(units, name=f"{name}_dense_2")(x)
    x = tf.keras.layers.Add(name=f"{name}_add")([shortcut, x])
    return tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)


def build_resnet34(input_size, class_count):
    """Build a ResNet-34-style MLP: 1 + (16 residual blocks * 2) + 1 layers."""
    inputs = tf.keras.Input(shape=(input_size,), name="tfidf_features")
    x = tf.keras.layers.Dense(128, activation="relu", name="input_projection")(inputs)
    x = tf.keras.layers.Dropout(0.3, name="input_dropout")(x)

    # The ResNet-34 block layout is 3 + 4 + 6 + 3 = 16 blocks.
    for block_number in range(1, 17):
        x = residual_block(x, 128, 0.2, f"residual_block_{block_number}")

    outputs = tf.keras.layers.Dense(
        class_count, activation="softmax", name="intent_output"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="resnet34_intent_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class ResNet34TextClassifier:
    """A scikit-learn-compatible interface around the TensorFlow ResNet-34 model."""

    def __init__(self, epochs=40, batch_size=16):
        self.epochs = epochs
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.label_encoder = LabelEncoder()

    def fit(self, texts, labels):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

        features = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.classes_ = self.label_encoder.classes_
        target = tf.keras.utils.to_categorical(encoded_labels, len(self.classes_))

        self.model = build_resnet34(features.shape[1], len(self.classes_))
        self.model.fit(
            features,
            target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self

    def predict_proba(self, texts):
        features = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self.model.predict(features, verbose=0)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        return self.classes_[np.argmax(probabilities, axis=1)]


def build_models(texts, labels):
    """Train all three required classifiers on the complete FAQ dataset."""
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
        ]),
        "ResNet-34 Neural Network": ResNet34TextClassifier(),
    }


@st.cache_resource
def train_models():
    intents = load_intents()
    texts, labels = make_examples(intents)
    models = build_models(texts, labels)
    for model in models.values():
        model.fit(texts, labels)
    return models


@st.cache_data
def compare_models():
    """A repeatable held-out test for the presentation; final chat models use all data."""

    intents = load_intents()
    texts, labels = make_examples(intents)

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    scores = {}

    for name, model in build_models(texts, labels).items():
        model.fit(x_train, y_train)

        prediction = model.predict(x_test)

        scores[name] = {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                prediction,
                average="weighted",
                zero_division=0
            )
        }

    return scores


def answer(question, model_name, intents):
    model = train_models()[model_name]
    probabilities = model.predict_proba([question])[0]
    confidence = float(probabilities.max())
    if confidence < 0.35:
        return FALLBACK, confidence, None
    tag = model.classes_[probabilities.argmax()]
    intent = next(item for item in intents if item["tag"] == tag)
    text = random.choice(intent["responses"])
    if intent.get("source_url"):
        text += f"\n\nMore information: {intent['source_url']}"
    return text, confidence, tag


st.set_page_config(page_title="TAR UMT FAQ Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 TAR UMT FAQ Chatbot")
st.caption("Ask about programmes, admissions, fees, campus, library, intakes and scholarships.")

intents = load_intents()
with st.sidebar:
    st.header("Model Comparison")
    st.write("Three required models are trained from the same FAQ patterns.")

    try:
        scores = compare_models()

        df = pd.DataFrame(scores).T

        st.dataframe(
            df.style.format("{:.2%}"),
            width="stretch"
        )

        best_model = max(
            scores,
            key=lambda x: scores[x]["F1 Score"]
        )

        st.success(
            f"Best Model: {best_model}"
        )

    except ValueError:
        best_model = "Logistic Regression"
        st.info("Not enough examples per class to produce a stratified test split.")

    selected_model = st.selectbox(
        "Chat Model",
        list(train_models()),
        index=list(train_models()).index(best_model)
    )

    st.caption(
        "Evaluation is based on a held-out test dataset."
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! What would you like to know about TAR UMT?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    reply, confidence, _ = answer(prompt, selected_model, intents)
    with st.chat_message("assistant"):
        st.markdown(reply)
        with st.expander("Prediction details"):
            st.caption(f"Model: {selected_model} · confidence: {confidence:.0%}")
    st.session_state.messages.append({"role": "assistant", "content": reply})
