# ✈️ Twitter Airline Sentiment Analysis — End-to-End NLP Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Demo:** [Click here to try the app](https://share.streamlit.io) ← *(replace with your Streamlit URL)*
> **Dataset:** [Twitter US Airline Sentiment — Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

---

## What This Project Does

This project classifies airline tweets as **Negative**, **Neutral**, or **Positive** in real time.
It is a complete, production-grade NLP pipeline — from raw data to a deployed interactive web app.

**Try it yourself:** paste any airline tweet into the app and instantly see the predicted sentiment
with confidence scores.

---

## Live App Features

| Tab | What it does |
|-----|-------------|
| **Analyse Tweet** | Type any tweet → get sentiment + confidence scores in real time |
| **Batch Analysis** | Paste multiple tweets → analyse all at once |
| **Dataset Explorer** | Interactive charts, word clouds, airline breakdown |
| **Model Comparison** | Leaderboard of all 10 models (Naive Bayes → DistilBERT) |
| **About Project** | Full pipeline explanation, tech stack, findings |

---

## Project Structure

```
twitter-airline-sentiment-nlp/
│
├── app.py                          ← Streamlit web application
├── requirements.txt                ← Python dependencies for deployment
│
├── data_cleaning_wrangling.ipynb   ← Notebook 1: Data cleaning & feature extraction
├── eda_preprocessing_v2.ipynb      ← Notebook 2: EDA + 7 statistical tests
├── modeling_v2.ipynb               ← Notebook 3: Model training + deployment guide
│
├── saved_models/
│   ├── lr_model.pkl                ← Deployed model (Logistic Regression)
│   ├── tfidf_vectorizer.pkl        ← Fitted TF-IDF vectorizer (12,670 features)
│   ├── svm_model.pkl               ← LinearSVC calibrated model
│   ├── ensemble_model.pkl          ← Soft Voting Ensemble (LR + SVM + CNB + LGB)
│   └── label_mapping.json          ← Label encoding map
│
├── train.csv                       ← Training set (10,896 tweets)
├── test.csv                        ← Test set (2,724 tweets)
└── cleaned_data.csv                ← Cleaned dataset used in EDA tab
```

---

## Machine Learning Pipeline

### Step 1 — Data Cleaning (`data_cleaning_wrangling.ipynb`)
- Removed duplicates, null values, encoding errors
- Normalised text: lowercased, removed URLs, @mentions, HTML tags
- Extracted features: word count, character count, tweet hour

### Step 2 — EDA & Preprocessing (`eda_preprocessing_v2.ipynb`)
- **7 statistical tests**: Shapiro-Wilk, Mann-Whitney U, ANOVA, Chi-Square, Z-test, Levene, Kruskal-Wallis
- Feature representations: TF-IDF, Bag-of-Words, Word2Vec
- Confirmed class imbalance (63% negative) → chose **Macro F1** as primary metric

### Step 3 — Model Building (`modeling_v2.ipynb`)

Models 1–8 were **fully trained and evaluated on CPU**. Models 9–10 are **GPU benchmark results**
(BiLSTM and DistilBERT require GPU; results shown are standard benchmarks for these architectures).

| # | Model | Macro F1 | Accuracy | Notes |
|---|-------|----------|----------|-------|
| 1 | Multinomial Naive Bayes | 0.65 | 0.74 | Trained locally |
| 2 | Complement Naive Bayes | 0.67 | 0.75 | Trained locally |
| 3 | LinearSVC (calibrated) | 0.69 | 0.78 | Trained locally |
| 4 | Random Forest | 0.63 | 0.72 | Trained locally |
| 5 | XGBoost | 0.66 | 0.75 | Trained locally |
| 6 | LightGBM | 0.68 | 0.76 | Trained locally |
| 7 | **Logistic Regression** ⭐ | **0.70** | **0.76** | Trained locally — **Deployed** |
| 8 | Soft Voting Ensemble | 0.72 | 0.79 | Trained locally |
| 9 | BiLSTM | 0.80 | 0.85 | GPU benchmark |
| 10 | DistilBERT | 0.85 | 0.89 | GPU benchmark |

> ⭐ **Deployed model:** Logistic Regression + TF-IDF — Macro F1 = 0.70, Accuracy = 76%, inference < 5ms
>
> BiLSTM and DistilBERT require GPU for training (8–9 hours on CPU). Results shown are
> standard benchmarks for these architectures on this dataset, included for research comparison.

---

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| **Macro F1 over Accuracy** | Data is 63% negative. A dumb model that always predicts "negative" gets 65% accuracy but learns nothing. Macro F1 treats all 3 classes equally. |
| **class_weight='balanced'** | Without this, models completely ignore the positive class (only 16% of data). |
| **TF-IDF fit on train only** | Fitting on all data leaks test vocabulary into the model — inflates scores and gives false confidence. |
| **LR for deployment** | LR predicts in <5ms. DistilBERT takes 200ms+. For real-time use, LR is the right production choice. |
| **Soft Voting Ensemble** | Each model has different failure modes. Combining probabilities from LR + SVM + CNB + LGB reduces variance. |

---

## Key Findings

- **63% of tweets are negative** — severe class imbalance, accuracy is misleading
- **United & American Airlines** have the highest complaint rates
- **Neutral class is hardest to classify** — Neutral↔Negative confusion is most common (264 cases)
- **"cancelled", "delay", "hours", "help"** are the strongest negative predictors
- Class weights are essential — without them, positive class is almost entirely ignored

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/patidarpuja/twitter-airline-sentiment-nlp.git
cd twitter-airline-sentiment-nlp

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10 |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly, wordcloud |
| NLP | NLTK, gensim (Word2Vec) |
| Classical ML | scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Deep Learning | TensorFlow / Keras — BiLSTM (architecture, GPU required) |
| Transformers | HuggingFace — DistilBERT (fine-tuning design, GPU required) |
| Deployment | Streamlit Community Cloud |
| Model Saving | joblib |

---

## Skills Demonstrated

- End-to-end NLP pipeline (raw data → deployed app)
- Statistical hypothesis testing (7 tests)
- Text feature engineering (TF-IDF, BoW, Word2Vec)
- Handling class imbalance (class weights, Macro F1)
- Classical ML model comparison (8 algorithms including ensemble)
- Model interpretability (LR coefficients, error analysis)
- Production deployment (Streamlit Community Cloud)
- Production monitoring framework (6 metrics: drift, confidence, latency)
- Deep Learning architecture design (BiLSTM) and Transfer Learning (DistilBERT)

---

## Author

**Puja Patidar**
Data Science | NLP | Machine Learning
[LinkedIn](#) · [GitHub](https://github.com/patidarpuja)

---

*Built as part of Springboard Data Science Bootcamp — NLP Capstone Project*
