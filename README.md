# Machine Learning | NLP | Sentiment Analysis Project
# On the Study of Resource-Efficient Sentiment Analysis in Social Media  
## Leveraging Emoji, Hashtags, and Random Forest Classifier

---

## 📌 Project Overview

This project presents a **resource-efficient approach to sentiment analysis** for social media platforms by leveraging textual content, emojis, and hashtags.  
The objective is to build a computationally efficient yet accurate **multi-sentiment classification system** using a traditional machine learning algorithm — **Random Forest Classifier**.

Unlike deep learning models that require heavy computational resources, this study demonstrates that optimized machine learning techniques combined with intelligent feature engineering can achieve strong performance while remaining lightweight and scalable.

---

## 🎯 Objectives

- Perform multi-class sentiment classification on social media data
- Incorporate **emoji-based sentiment understanding**
- Utilize **hashtags as contextual sentiment indicators**
- Develop a resource-efficient ML pipeline
- Evaluate model performance using multiple evaluation metrics

---

## 😊 Sentiment Categories

The model classifies input text into the following sentiment classes:

- Positive
- Negative
- Neutral
- Angry
- Joyful
- Sad
- Fearful
- Surprised
- Love
- Sarcastic

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Text normalization
- Removal of noise and special characters
- Stopword removal
- Tokenization and lemmatization
- Emoji extraction and sentiment mapping
- Hashtag identification and processing

### 2️⃣ Feature Engineering
- TF-IDF Vectorization
- Emoji sentiment scoring
- Hashtag frequency encoding
- Feature combination for improved representation

### 3️⃣ Model Development
- Algorithm: **Random Forest Classifier**
- Training and validation split
- Hyperparameter optimization
- Efficient feature selection

### 4️⃣ Model Evaluation
Performance evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## 🛠️ Technologies & Libraries

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK / SpaCy
- Emoji processing libraries

---

## 📊 Results & Findings

- Achieved high multi-class classification performance
- Emoji and hashtag integration significantly improved sentiment prediction
- Random Forest provided strong accuracy with reduced computational cost
- Demonstrated effectiveness of traditional ML models for NLP tasks

---

## 🚀 Key Contributions

- Multi-sentiment classification beyond binary sentiment analysis
- Integration of emoji intelligence into NLP workflow
- Hashtag-aware sentiment modeling
- Resource-efficient alternative to deep learning models
- Practical implementation suitable for real-world deployment

---

## 📁 Project Structure
Resource-Efficient-Sentiment-Analysis
│
├── dataset/ # Input datasets (sample data)
│
├── notebooks/ # Jupyter notebooks for experimentation
│ └── sentiment_analysis.ipynb
│
├── src/ # Source code files
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ └── evaluation.py
│
├── model/ # Saved trained models
│ └── random_forest_model.pkl
│
├── outputs/ # Generated plots and results
│ ├── confusion_matrix.png
│ ├── accuracy_plot.png
│ └── roc_curve.png
│
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── main.py # Main execution script


---

## ▶️ How to Run the Project

### 1. Clone Repository
git clone https://github.com/yourusername/resource-efficient-sentiment-analysis.git

cd resource-efficient-sentiment-analysis


### 2. Install Dependencies
pip install -r requirements.txt


### 3. Run the Model
python main.py


---

## 📈 Future Scope

- Real-time sentiment monitoring dashboard
- Deployment using Flask or FastAPI
- Integration with Twitter/X or Instagram APIs
- Deep learning comparison (LSTM, BERT)
- Multilingual sentiment analysis support

---

## 👩‍💻 Author

**Rudrani Mukherjee**  
B.Tech – Computer Science & Engineering (Data Science)  
Machine Learning | NLP | Data Analytics Enthusiast

---

## 📜 License

This project is developed for academic and research purposes.

---

## ⭐ Acknowledgement

This work explores efficient machine learning solutions for sentiment understanding in modern social media communication environments.
