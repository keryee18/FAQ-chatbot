#Logistic Regression Model
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Download required text tools
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# 1. Load intents.json file
try:
    with open("intents.json", "r") as file:
        data = json.load(file)
    print("Successfully connected to your custom intents.json file!\n")
except FileNotFoundError:
    print("ERROR: 'intents.json' not found. Please upload your file.")
    raise

X = []  # Patterns
y = []  # Tags

# 2. Preprocess Text
# Breaks to single word, convert all to lowercase, reduce words to base form
def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words])

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(preprocess(pattern))
        y.append(intent["tag"])

# 3. Build & Train the Logistic Regression Model
# convert: preprocessed patterns (x) -> numerical feature vectors
# TF-IDF (Term Frequency-Inverse Document Frequency) - statistical measure that evaluates how relevant a word is to a document in a collection of documents
# Logistic regression - linear model used for binary or multi-class classification
model = make_pipeline( # make_pipeline -> chain TF-IDF & logistic regression
    TfidfVectorizer(),
    LogisticRegression(random_state=42)
)
model.fit(X, y)
print("--- Logistic Regression Model Training Complete (Runs instantly) ---")

# 4. Chat Interface Function
def get_lr_response(user_input):
    cleaned_input = preprocess(user_input) # clean user input
    probabilities = model.predict_proba([cleaned_input])[0] # predict probability of user input belonging to each intent
    max_prob_index = probabilities.argmax()
    max_prob = probabilities[max_prob_index]

    if max_prob < 0.25:
        return "I'm sorry, I don't understand that question."

    predicted_tag = model.classes_[max_prob_index]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I am unsure how to answer that."

# 5. Live Chat Loop
print("Logistic Regression Bot is ready! Type 'quit' to exit.\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break
    response = get_lr_response(message)
    print(f"Bot: {response}\n")
