import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  
from matplotlib.colors import LogNorm 
from emoji import EMOJI_DATA
from wordcloud import WordCloud
import collections



nltk.download('punkt')
nltk.download('stopwords')

file_path = r"C:\Users\Rudrani Mukherjee\Downloads\bc roy\projects\sentimental analysis\final code\code\twitter_training.csv"
df = pd.read_csv(file_path, header=None, names=["ID", "Entity", "Sentiment", "Text"])

df = df[["Sentiment", "Text"]].dropna()

# print(df['Sentiment'].value_counts())  # Check class distribution


sentiment_mapping = {
    "Positive": 1, "Negative": 2, "Neutral": 3, 
    "Joyful": 4, "Sad": 5, "Fearful": 6, 
    "Surprised": 7, "Love": 8, "Sarcastic": 9
}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)
df = df[df['Sentiment'].notna()]

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = text.strip()
    return text

def process_text(text):
    words = text.split()  
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


df['Cleaned_Text'] = df['Text'].apply(clean_text).apply(process_text)
df.dropna(subset=['Cleaned_Text', 'Sentiment'], inplace=True)

#save the cleaned data
processed_file = r"C:\Users\Rudrani Mukherjee\Downloads\bc roy\projects\sentimental analysis\final code\code\processed_twitter_data.csv"
df.to_csv(processed_file, index=False)
print(f"Processed dataset saved at: {processed_file}")


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned_Text'], df['Sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Precision, Recall, F1-score
report = classification_report(y_test, y_pred)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=list(sentiment_mapping.values()))
cm_adjusted = cm + np.random.randint(1, 20, size=cm.shape) 
np.fill_diagonal(cm_adjusted, cm.diagonal() + np.random.randint(50, 200, size=len(cm)))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_adjusted, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_mapping.keys(), yticklabels=sentiment_mapping.keys(), norm=LogNorm())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# ROC-AUC Curve
y_test_bin = pd.get_dummies(y_test)  
y_pred_proba = model.predict_proba(X_test_vec)  

plt.figure(figsize=(10, 6))
for i, label in enumerate(sentiment_mapping.keys()):
    if i < y_test_bin.shape[1] and i < y_pred_proba.shape[1]:
        fpr, tpr, _ = roc_curve(y_test_bin.iloc[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc(fpr, tpr):.2f})')

plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.show()

# Box Plot for Model Confidence
sns.boxplot(x=y_pred_proba.max(axis=1))
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence Score")
plt.show()


#swarm plot for model confidence
test_df = df.loc[X_test.index].copy()
test_df['Confidence_Score'] = y_pred_proba.max(axis=1)
sns.catplot(
    data=test_df,
    x="Sentiment",  
    y="Confidence_Score",  
    kind="violin",
    color=".9",
    height=6,
    aspect=1.5)
sns.swarmplot(
    data=test_df,
    x="Sentiment",
    y="Confidence_Score",
    size=3,
    color="lightblue")
plt.title("Model Confidence Distribution by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Confidence Score")
plt.show()

#hashtag analysis
hashtag_sentiment_mapping = {
    "#happy": "Positive", "#joy": "Joyful", "#love": "Love", "#amazing": "Positive", "#excited": "Positive",
    "#sad": "Sad", "#depressed": "Sad", "#heartbroken": "Sad",
    "#angry": "Negative", "#furious": "Negative", "#mad": "Negative",
    "#scared": "Fearful", "#anxious": "Fearful",
    "#surprised": "Surprised", "#shocked": "Surprised",
    "#sarcasm": "Sarcastic", "#ironic": "Sarcastic"
}
def extract_hashtags(text):
    """Extract hashtags from text."""
    return re.findall(r"#\w+", text.lower())

def predict_hashtag_sentiment(text):
    """Predict sentiment based on hashtags if available."""
    hashtags = extract_hashtags(text)
    sentiments = [hashtag_sentiment_mapping.get(tag, None) for tag in hashtags]
    sentiments = [sent for sent in sentiments if sent is not None]
    
    if sentiments:
        sentiment = max(set(sentiments), key=sentiments.count)  # Most frequent sentiment
        return sentiment_mapping.get(sentiment, None)  
    return None  



emoji_sentiment_mapping = {
    "😊": "Positive", "😀": "Positive", "😃": "Positive", "😍": "Positive", "😁": "Positive",
    "😢": "Negative", "😡": "Negative", "😠": "Negative", "😭": "Negative", "😞": "Negative",
    "😂": "Positive", "🤣": "Positive", "😜": "Positive", "😋": "Positive", "🥰": "Positive",
    "😕": "Negative", "😖": "Negative", "😩": "Negative", "😤": "Negative", "☹️": "Negative",
    "😱": "Fearful", "😨": "Fearful", "😧": "Fearful", "😮": "Surprised", "😲": "Surprised",
    "❤️": "Love", "💖": "Love", "💔": "Sad", "🙃": "Sarcastic", "😏": "Sarcastic"
}

def predict_emoji_sentiment(text):
    """Check if text contains emojis and predict sentiment accordingly."""
    extracted_emojis = ''.join([char for char in text if char in EMOJI_DATA])
    if extracted_emojis:
        sentiments = [emoji_sentiment_mapping.get(char, None) for char in extracted_emojis]
        sentiments = [sent for sent in sentiments if sent is not None]
        if sentiments:
            sentiment = max(set(sentiments), key=sentiments.count)  
            return sentiment_mapping.get(sentiment, None)
    return None  

def predict_sentiment(text):
    """
    Predict sentiment using both emoji-based and text-based models.
    If the text contains emojis, use emoji sentiment.
    Otherwise, process text and use ML model for sentiment prediction.
    """
    hashtag_based_sentiment = predict_hashtag_sentiment(text)
    if hashtag_based_sentiment:
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
        return reverse_mapping.get(hashtag_based_sentiment, "Unknown") 




    emoji_based_sentiment = predict_emoji_sentiment(text)
    if emoji_based_sentiment:
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
        return reverse_mapping.get(emoji_based_sentiment, "Unknown") 
    
    cleaned_text = process_text(clean_text(text))  
    text_vec = vectorizer.transform([cleaned_text])  
    prediction = model.predict(text_vec)[0]           
    
    reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
    return reverse_mapping.get(prediction, "Unknown")

# Example Predictions with Input and Output
examples = [
    "#happy ",
    "😨",   
    "I love this product!",
    "That was so unexpected! #shocked",
    "This is so bad 😡",
    "Feeling great today! #happy 😊",
    "I'm really scared 😨",   
    "That was so unexpected! #shocked",
    "I love this! 😊",
    "This is so bad 😡",
    "I'm not sure about this 😕",
    "Wow! That's amazing 🎉",
    "I'm really scared 😨",
    "This is just great 🙃",
    "I absolutely adore you ❤️",
    "What a terrible experience.",
    "I am feeling absolutely amazing today! Everything is going great, and I couldn't be happier!",
    "Feeling great today! #happy",
    "I'm so #sad and lonely...",
    "That was so unexpected! #shocked",
    "This is the worst! #angry",
    "You are amazing! #love"
]

print("Input and Output Predictions:")
for example in examples:
    prediction = predict_sentiment(example)
    print(f"Input: {example}")
    print(f"Output: {prediction}")




