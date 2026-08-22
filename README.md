# Project Title: Chatbot Development (TAR UMT FAQ Chatbot)
## Group Number: 
## Members: Chong Ker Yee, Lai Ming Tong, Low Qian Tong
## Brief Purpose:
* The **TAR UMT FAQ Chatbot** is an AI-driven conversational web application designed to automatically answer student          inquiries regarding programmes, admissions, tuition fees, library services, intakes, scholarships, and campus locations.
* By leveraging Natural Language Processing (NLP) and Machine Learning (ML), the system evaluates and compares three           classification approaches—**Logistic Regression**, **Random Forest**, and a **Deep Residual MLP (ResNet-34)**—to classify    user questions accurately, enforce confidence-based fallback rules, and provide instant administrative support.
## Main Prototype Functions:
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
## Programming Language:
**Language**: Python 3.10+
**Environment**: Anaconda / venv

### Core Frameworks & Tool Versions
| Category | Tool / Library | Recommended Version |
| :--- | :--- | :--- |
| **Web UI** | Streamlit | ^1.28.0 |
| **Deep Learning** | TensorFlow / Keras | ^2.15.0 |
| **Machine Learning** | scikit-learn | ^1.3.0 |
| **NLP** | NLTK | ^3.8.1 |
| **Data Processing** | NumPy / Pandas | ^1.24.3 / ^2.1.0 |
## Support Operating System: 
* **Microsoft Windows**: Windows 10 / Windows 11 (64-bit) — Tested
* **macOS**: macOS 12 (Monterey) or later (Intel & Apple Silicon M-series) — Supported
* **Linux**: Ubuntu 20.04 LTS / 22.04 LTS or equivalent distributions — Supported
## Execution Environment: 
* **Python Runtime**: Python `3.10.x` or higher (64-bit)
* **Virtual Environment**: Python `venv` or Anaconda (`conda`) environment recommended
* **Interactive CLI / Web Terminal**: Any standard terminal emulator (Windows PowerShell, Command Prompt, macOS Terminal, or VS Code Integrated Terminal)
* **Web Browser Interface**: Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari (for accessing the Streamlit Web Application)
## Installation & Running commands: 
### 1. Clone the Repository
Clone the project repository from GitHub:
```git clone https://github.com/keryee18/FAQ-chatbot.git```

### 2. Install Dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Web Prototype
```streamlit run app.py```
## Datasets and Trained-model setup: 
### 1. Dataset (```intents.json```)
**Source & Origin**: The dataset was custom-built specifically for this project to handle common student inquiries at Tunku Abdul Rahman University of Management and Technology (TAR UMT).
**Structure**: Formatted as a JSON object containing structured intent categories. Each category includes:
  * ```tag```: The intent label (e.g., admissions, tuition fees, library, campus locations).
  * ```patterns```: Sample user queries and variation phrases used to train the classifier.
  * ```responses```: Predefined answer templates.
  * ```source_url```: Official TAR UMT website URLs dynamically appended to answers.
**Pre-processing Requirements**: No manual download or prior dataset preparation is required. Data loading, tokenization, lemmatization (via NLTK), and feature extraction occur automatically at application startup.

---

### 2. Trained-Model Setup
**Automated Initialization**: When you execute ```streamlit run app.py```, the application automatically loads ```intents.json```, preprocesses the text, and trains all three models (**Logistic Regression**, **Random Forest**, and **Deep Residual Multilayer Perceptron (ResNet-34 Style)**) in memory within seconds.
**Re-training / Modification**: If you modify or add patterns to ```intents.json```, simply refresh the Streamlit web application to automatically retrain the models with the updated dataset.
## Test-input instructions and expected outputs: 
You can refer to ```intents.json```.
### Operating system tested.
### Python, Java, Node.js or other runtime version.
### Required software, such as VS Code, Anaconda, Jupyter or Google Colab.
### Library/package names and versions.
### Internet, database, API or external-service requirements.
### CPU, RAM or GPU requirements, if relevant.
### Approximate installation, model-loading, training and prediction time.
