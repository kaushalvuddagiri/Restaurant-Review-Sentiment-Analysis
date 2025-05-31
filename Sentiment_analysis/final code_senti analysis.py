import pandas as pd
import re
import nltk
import pickle
import joblib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Load and preprocess the dataset ---
nltk.download('stopwords')

def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')  # Keep 'not' for sentiment
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return ' '.join([ps.stem(word) for word in words if word not in stop_words])

# Load data
dataset_path = r"E:\ECE BOOKS\future_CSE PROJECTS\hex softwares\Sentiment_analysis\fromdrive\a2_RestaurantReviews_FreshDump.tsv"
dataset = pd.read_csv(dataset_path, delimiter='\t', quoting=3)

# Preprocess reviews
corpus = [clean_text(review) for review in dataset['Review'][:100]]

# --- Load vectorizer and model ---
cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
classifier = joblib.load('c2_Classifier_Sentiment_Model')

# --- Transform and predict ---
X = cv.transform(corpus).toarray()
y_pred = classifier.predict(X)

# Add predictions to dataset
dataset['predicted_label'] = y_pred.tolist()

# Save results
output_path = r"E:\ECE BOOKS\future_CSE PROJECTS\hex softwares\Sentiment_analysis\c4_Predicted_Sentiments_Fresh_Dump.tsv"
dataset.to_csv(output_path, sep='\t', encoding='utf-8', index=False)

# --- Visualize sentiment counts ---
label_counts = dataset['predicted_label'].value_counts()
labels = ['Negative', 'Positive']
counts = [label_counts.get(0, 0), label_counts.get(1, 0)]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, counts, color=['red', 'green'])

# Add count labels
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, int(bar.get_height()), ha='center')

plt.title('Sentiment Prediction Counts')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.ylim(0, max(counts) + 10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
