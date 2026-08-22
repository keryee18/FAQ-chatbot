#Project Title: Chatbot Development (TAR UMT FAQ Chatbot)
Group Number: 
Members: Chong Ker Yee, Lai Ming Tong, Low Qian Tong
Brief Purpose:
* The **TAR UMT FAQ Chatbot** is an AI-driven conversational web application designed to automatically answer student          inquiries regarding programmes, admissions, tuition fees, library services, intakes, scholarships, and campus locations.
* By leveraging Natural Language Processing (NLP) and Machine Learning (ML), the system evaluates and compares three           classification approaches—**Logistic Regression**, **Random Forest**, and a **Deep Residual MLP (ResNet-34)**—to classify    user questions accurately, enforce confidence-based fallback rules, and provide instant administrative support.
Main Prototype Functions:
### 1. Primary Entry Point (app.py)
The main entry point of the project that runs the Streamlit Web Application (streamlit run app.py).
**Model Comparison Engine**: Executes 5-fold cross-validation on startup to display real-time Accuracy, Precision, Recall, and F1-score comparisons across all models in the sidebar.
**Input Validation & Safety Rules**: Filters out non-topical generic query words and enforces a 35% confidence score threshold to trigger fallback responses when questions are unclear.
**Interactive Chat Interface**: Maintains model-specific session transcripts and dynamically appends official TAR UMT source URLs to answers.

### 2. Standalone CLI & Model Modules
These scripts allow standalone training, evaluation, and CLI-based chat testing for individual algorithms:
**logisticregression.py**: Implements a TF-IDF + Logistic Regression pipeline with balanced class weights and a interactive command-line interface.
**random_forest.py**: Implements a TF-IDF + Random Forest Classifier (300 estimators) pipeline with interactive command-line chat.
**deep_learning.py**: Builds, trains, and evaluates a 34-layer Deep Residual Neural Network (MLP) using Keras with Bag-of-Words feature extraction.
Programming Language:
Framework:
Important Tool Versions:
Support Operating System:
Execution Environment:
Installation commands:

<!-- run in terminal:
pip install -r requirements.txt
streamlit run app.py
-->

