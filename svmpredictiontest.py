import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('popular')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
import joblib
import re

np.random.seed(500)

# Load the saved model and vectorizer
loaded_model = joblib.load('model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

def combine_dead_bolt(tokens):
    combined_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == 'dead' and i + 1 < len(tokens) and tokens[i + 1] == 'bolt':
            combined_tokens.append('dead_bolt')
            i += 2
        else:
            combined_tokens.append(tokens[i])
            i += 1
    return combined_tokens

def remove_special_characters(text):
    text = re.sub(r'[()//\\\\-]', ' ', text)
    return text
# Load the new data
new_data = pd.read_excel(r"datafile")
new_data['LossDescription'].dropna(inplace=True)
new_data['LossDescription'] = new_data['LossDescription'].astype(str)  # Convert to string type
new_data['LossDescription'] = new_data['LossDescription'].apply(remove_special_characters)
new_data['LossDescription'] = [entry.lower() for entry in new_data['LossDescription']]
new_data['LossDescription'] = [word_tokenize(entry) for entry in new_data['LossDescription']]

# Preprocess the new data
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['X'] = wn.ADJ
tag_map['Y'] = wn.VERB
tag_map['Z'] = wn.ADV

for index, entry in enumerate(new_data['LossDescription']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') or word in ["in", "on", "no", "non"]:
            if word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
    new_data.loc[index, 'text_final'] = str(Final_words)

# Transform the new data using the loaded vectorizer
new_data_tfidf = loaded_vectorizer.transform(new_data['text_final'])

# Make predictions on the new data
predictions_new_data = loaded_model.predict(new_data_tfidf)

# Get the probability estimates for each class
probabilities = loaded_model.predict_proba(new_data_tfidf)

# Add the predictions to the new_data DataFrame
new_data['Prediction'] = predictions_new_data

# Add the confidence probabilities to the new_data DataFrame
new_data['Confidence'] = probabilities.max(axis=1)

# Save the DataFrame with predictions and confidence to a new Excel file
output_path = 'savepredictionsfile'
new_data.to_excel(output_path, index=False)
print("Predictions saved to:", output_path)

count_0 = np.count_nonzero(predictions_new_data == 0)
count_1 = np.count_nonzero(predictions_new_data == 1)

print("Number of 0 predictions:", count_0)
print("Number of 1 predictions:", count_1)
