"""
Filename: logistic_regression.py
Project: Chatbot Development
Author: Low Qian Tong
Description:   
    1. Loads 'intents.json' and preprocesses input text via NLTK (Tokenization & Lemmatization).
    2. Builds a Scikit-Learn Pipeline combining TF-IDF Vectorizer and Logistic Regression Classifier.
    3. Calculates prediction probabilities with a confidence threshold (0.25) and appends source URLs.
    4. Provides an interactive command-line interface (CLI) chat loop for user testing.
"""

#Logistic Regression Model
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

#Download required text tools
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

#Load intents.json file
try:
    with open("intents.json", "r") as file:
        data = json.load(file)
    print("Successfully connected to your custom intents.json file!\n")
except FileNotFoundError:
    print("ERROR: 'intents.json' not found. Please upload your file.")
    raise

X = []  #Patterns
y = []  #Tags

#Preprocess Text
#Breaks to single word, convert all to lowercase, reduce words to base form
def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words])

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(preprocess(pattern))
        y.append(intent["tag"])

"""Build & Train the Logistic Regression Model
convert: preprocessed patterns -> numerical feature vectors
TF-IDF (Term Frequency-Inverse Document Frequency) - statistical measure that evaluates how relevant a word is to a document in a collection of documents
Logistic regression - linear model used for binary or multi-class classification"""
model = make_pipeline( # make_pipeline -> chain TF-IDF & logistic regression
    TfidfVectorizer(),
    LogisticRegression(random_state=42)
)
model.fit(X, y)
print("--- Logistic Regression Model Training Complete (Runs instantly) ---")

#Chat Interface Function
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
            #Always choose a response first
            selected_response_text = random.choice(intent["responses"]) if intent["responses"] else ""

            response_parts = [selected_response_text]

            #If a source URL exists, add it to the parts
            if "source_url" in intent and intent["source_url"]:
                response_parts.append(f"For more information, visit: {intent['source_url']}")

            #Join the parts, only including non-empty ones
            final_response = "\n".join(filter(None, response_parts))
            if not final_response: # If even after combining, it's empty, provide a fallback
                 return "I am unsure how to answer that."

            return final_response
    return "I am unsure how to answer that."

#Live Chat Loop
print("Logistic Regression Bot is ready! Type 'quit' to exit.\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break
    response = get_lr_response(message)
    print(f"Bot: {response}\n")
