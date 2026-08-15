#Deep Learning Neutral Network Model
import json
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Add, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Download necessary NLTK components for processing text
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")

lemmatizer = WordNetLemmatizer()

# 1. Load the data
with open("intents.json") as file:
    data = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# 2. Preprocess the Data (Tokenization and mapping tags)
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize patterns (breaks sentences into words)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents
        documents.append((word_list, intent["tag"]))
        # Add to classes
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize (reduces words to their base form, change to lowercase) and clean words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# 3. Vectorization (Creating Bag-of-Words vectors for training)
# bag-of-words -> representation for each pattern (convert: text -> numerical vectors)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and split into features (X) and labels (Y)
# To ensure the model doesn't learn any unintended order/patterns present in the original dataset
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# 4. Build and Train a ResNet-style neural network
#
# This is a ResNet-34-style residual MLP, appropriate for Bag-of-Words feature
# vectors. It has 16 residual blocks with two dense layers each, plus the input
# projection and output classifier: 1 + (16 * 2) + 1 = 34 dense layers.
def residual_block(inputs, units, dropout_rate, name):
    """Apply two dense layers and add their output back to the shortcut."""
    shortcut = inputs
    x = Dense(units, activation="relu", name=f"{name}_dense_1")(inputs)
    x = Dropout(dropout_rate, name=f"{name}_dropout")(x)
    x = Dense(units, activation=None, name=f"{name}_dense_2")(x)
    x = Add(name=f"{name}_add")([shortcut, x])
    return Activation("relu", name=f"{name}_relu")(x)


inputs = Input(shape=(len(train_x[0]),), name="bag_of_words")
# Project the vocabulary-sized input to a fixed width so residual additions
# have matching shapes.
x = Dense(128, activation="relu", name="input_projection")(inputs)
x = Dropout(0.5, name="input_dropout")(x)

# ResNet-34 has 16 two-layer residual blocks (3 + 4 + 6 + 3). Keeping their
# width at 128 makes the skip connections valid for text feature vectors.
for block_number in range(1, 17):
    x = residual_block(
        x,
        units=128,
        dropout_rate=0.3,
        name=f"residual_block_{block_number}"
    )

outputs = Dense(len(train_y[0]), activation="softmax", name="intent_output")(x)
model = Model(inputs=inputs, outputs=outputs, name="resnet34_intent_classifier")

# Compile the model (multi-class classification)
model.compile(optimizer=Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
print("\n--- Training Started ---")
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=0)
print("--- Training Finished ---\n")

# 5. Functions to process user input during live chat
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Determine most likely intent fro user's message
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Provide appropriate response from `intents.json`
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I don't understand that question. Can you please rephrase?"
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            # Always choose a response first
            responses = i.get("responses", [])
            selected_response_text = random.choice(responses) if responses else ""

            response_parts = [selected_response_text]

            # If a source URL exists, add it to the parts
            if "source_url" in i and i["source_url"]:
                response_parts.append(f"For more information, visit: {i['source_url']}")

            # Join the parts, only including non-empty ones
            final_response = "\n".join(filter(None, response_parts))
            if not final_response: # If even after combining, it's empty, provide a fallback
                 return "I am unsure how to answer that."

            return final_response
    return "I am unsure how to answer that."

# 6. Live Interaction Chatbot loop (continues until user type "quit")
print("Bot is ready! Type 'quit' to exit the chat.\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break

    ints = predict_class(message)
    res = get_response(ints, data)
    print(f"Bot: {res}\n")
