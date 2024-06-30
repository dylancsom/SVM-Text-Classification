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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import re

np.random.seed(500)

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
    text = re.sub(r'[()//\\\\*-]', ' ', text)
    return text

Corpus = pd.read_excel(r"/content/drive/MyDrive/ProductionSVMModel/Data/trainingwithedits.xlsx")

num_rows = Corpus.shape[0]
print("Number of rows in the dataset:", num_rows)

Corpus['text'].dropna(inplace=True)

Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['X'] = wn.ADJ
tag_map['Y'] = wn.VERB
tag_map['Z'] = wn.ADV

for index, entry in enumerate(Corpus['text']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    entry = combine_dead_bolt(entry)
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') or word in ["in", "on", "no", "non"]:
            if word.isalpha() and word not in ["insured", "insured's", "claimant", "clmnt", "clmt", "cv", "ivd", "cvd", "tboned","tbone", "ov", "iv"]:  # Check if word is not "insured" or "insured's"
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
    Corpus.loc[index, 'text_final'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = train_test_split(Corpus['text_final'], Corpus['label'], test_size=0.2, stratify=Corpus['label'], random_state=35)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)

precision = precision_score(Test_Y, predictions_SVM)
recall = recall_score(Test_Y, predictions_SVM)
f1 = f1_score(Test_Y, predictions_SVM)
conf_matrix = confusion_matrix(Test_Y, predictions_SVM)
report = classification_report(Test_Y, predictions_SVM)
false_positives = conf_matrix[1][0]
false_negatives = conf_matrix[0][1]

false_positives_indices = [index for index, (true_label, pred_label) in enumerate(zip(Test_Y, predictions_SVM)) if true_label == 0 and pred_label == 1]
false_negatives_indices = [index for index, (true_label, pred_label) in enumerate(zip(Test_Y, predictions_SVM)) if true_label == 1 and pred_label == 0]

false_positives_indices = [index for index, (true_label, pred_label) in enumerate(zip(Test_Y, predictions_SVM)) if true_label == 0 and pred_label == 1]
false_negatives_indices = [index for index, (true_label, pred_label) in enumerate(zip(Test_Y, predictions_SVM)) if true_label == 1 and pred_label == 0]

# Get the confidence scores for the test samples
confidence_scores = SVM.predict_proba(Test_X_Tfidf)

joblib.dump(SVM, '/content/drive/MyDrive/ProductionSVMModel/ModelDumps/mach2t5_svm.joblib')
joblib.dump(Tfidf_vect, '/content/drive/MyDrive/ProductionSVMModel/ModelDumps/mach2t5_tfidf_vectorizer.joblib')

print("SVM Accuracy: ", accuracy_score(predictions_SVM, Test_Y) * 100)
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

print("False Positives:")
for idx in false_positives_indices:
    print("Text:", Corpus['text_final'].iloc[idx])
    print("Confidence Score:", confidence_scores[idx][1])  # Confidence score for the positive class (1)
    print()

print("False Negatives:")
for idx in false_negatives_indices:
    print("Text:", Corpus['text_final'].iloc[idx])
    print("Confidence Score:", confidence_scores[idx][0])  # Confidence score for the negative class (0)
    print()