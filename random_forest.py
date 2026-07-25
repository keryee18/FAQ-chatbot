#Random Forest Model
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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
    print("ERROR: 'intents.json' not found. Please upload your file to the left sidebar menu in Google Colab.")
    raise

X = []  # Patterns (inputs)
y = []  # Tags (labels)

# 2. Preprocess Text
# Breaks to single word, convert all to lowercase, reduce words to base form
def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words])

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(preprocess(pattern))
        y.append(intent["tag"])

# 3. Build & Train the Random Forest Model
# We couple the TF-IDF vectorizer with a Random Forest Classifier
model = make_pipeline( # combine TF-IDF & random forest
    TfidfVectorizer(), # converts textual patterns -> numerical feature vectors

    # constructs a multitude of decision trees during training
    # outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees
    RandomForestClassifier(n_estimators=100, random_state=42) # 100 decision trees
    # 42 - popular number among programmers & data scientists (answer to the ultimate question of life, the universe & everything)
    # ensure that if anyone run the code again, the model will use the exact same random selections -> same result
)
model.fit(X, y)
print("--- Random Forest Model Training Complete (Runs instantly) ---")

# 4. Chat Interface Function
def get_rf_response(user_input):
    # clean user input
    cleaned_input = preprocess(user_input)

    # Predict probabilities for each intent class
    probabilities = model.predict_proba([cleaned_input])[0]
    max_prob_index = probabilities.argmax()
    max_prob = probabilities[max_prob_index]

    # Confidence threshold to filter out unrelated inputs
    if max_prob < 0.25:
        return "I'm sorry, I don't understand that question."

    predicted_tag = model.classes_[max_prob_index]

    # Retrieve matching response
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "I am unsure how to answer that."

# 5. Live Chat Loop
print("Random Forest Bot is ready! Type 'quit' to exit.\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break
    response = get_rf_response(message)
    print(f"Bot: {response}\n")
