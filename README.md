# News-Detector

A Machine Learning-based Fake News Detection system that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and supervised learning models.

The project uses **TF-IDF Vectorization** along with multiple machine learning algorithms such as:

- Logistic Regression
- Passive Aggressive Classifier

The application also includes a simple and interactive **Streamlit UI** for real-time news prediction.

---

# Features

- Detects Fake and Real news articles
- NLP preprocessing and text cleaning
- TF-IDF text vectorization
- Multiple ML model comparison
- Streamlit-based web interface
- Real-time prediction system
- High accuracy text classification

---

# Technologies Used

## Programming Language

- Python

## Libraries & Frameworks

- Pandas
- Scikit-learn
- Streamlit

## Machine Learning Concepts

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Text Classification
- Supervised Learning

---

# Project Structure

```bash
News-Detector/
│
├── app.py
├── fake.csv
├── true.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses two datasets:

- `fake.csv` → Fake news articles
- `true.csv` → Real news articles

Each article is labeled and used for supervised training.

---

# Data Preprocessing

The following preprocessing steps are applied:

- Lowercase conversion
- Removing special characters
- Removing stop words
- TF-IDF feature extraction
- Data shuffling
- Null value handling
- Duplicate removal

---

# Models Used

## Logistic Regression

A strong baseline model for text classification tasks.

## Passive Aggressive Classifier

An online learning algorithm that performs efficiently on large-scale text datasets and often gives better performance for fake news detection.

---

# Political Influence & Dataset Bias

This project demonstrates how Machine Learning models can sometimes become influenced by patterns present in the training dataset.

Since many public fake-news datasets contain a large amount of political news from specific regions (especially US politics), the model may:

- Perform better on political news articles
- Misclassify scientific or regional news
- Learn writing patterns instead of factual truth

### Example

- NATO or US political articles may be classified more accurately
- ISRO or regional Indian news may occasionally be misclassified

This highlights an important challenge in Fake News Detection systems:

> Machine Learning models do not truly verify facts — they learn statistical language patterns from training data.

Future improvements can include:

- Larger and more diverse datasets
- BERT/Transformer-based models
- Fact-checking APIs
- Source credibility analysis

---

# Model Accuracy

The project achieves high classification accuracy using TF-IDF and Passive Aggressive Classifier.

### Example Results

- Logistic Regression Accuracy: ~99%
- Passive Aggressive Accuracy: ~99%

> Accuracy may vary depending on dataset quality and preprocessing.

---

# Streamlit UI

The project includes a Streamlit-based interface where users can:

- Paste news articles
- Select prediction model
- Predict Real or Fake news instantly

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/News-Detector.git
```

Move into the project directory:

```bash
cd News-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Create a `requirements.txt` file with the following:

```txt
pandas
scikit-learn
streamlit
```

---

# Run the Project

```bash
streamlit run app.py
```

---

# Example Predictions

### Real News

```text
The United States President met with NATO leaders in Brussels to discuss regional security and economic cooperation.
```

### Fake News

```text
Scientists confirmed that eating chocolate pizza every day can permanently increase human IQ by 300 percent.
```

---

# Future Improvements

- Deep Learning models (LSTM/BERT)
- Better multilingual support
- Real-time news verification APIs
- News source reliability scoring
- Deployment on Streamlit Cloud

---

# Author

## Arpit Shirbhate

Aspiring Data Scientist & Machine Learning Enthusiast passionate about NLP, AI, and building real-world ML applications.
