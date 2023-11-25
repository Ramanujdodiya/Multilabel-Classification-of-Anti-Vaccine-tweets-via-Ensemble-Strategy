def distilbert_predict_sentence(sentence):
    # Tokenize the input sentence
    inputs = distilbert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        logits = outputs.logits    
    probabilities = torch.sigmoid(logits)
    threshold = 0.7
    predicted_labels_index = (probabilities > threshold).numpy()
    return predicted_labels_index, probabilities.numpy()
def xlnet_predict_sentence(sentence):
    # Tokenize the input sentence
    inputs = xlnet_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = xlnet_model(**inputs)
        logits = outputs.logits    
    probabilities = torch.sigmoid(logits)
    threshold = 0.7
    predicted_labels_index = (probabilities > threshold).numpy()
    return predicted_labels_index, probabilities.numpy()
def ctbert_predict_sentence(sentence):
    # Tokenize the input sentence
    inputs = ctbert_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = ctbert_model(**inputs)
        logits = outputs.logits    
    probabilities = torch.sigmoid(logits)
    threshold = 0.7
    predicted_labels_index = (probabilities > threshold).numpy()
    return predicted_labels_index, probabilities.numpy()
def ensemble_predict_text(sentence):
    # Get individual model predictions
    distilbert_labels, _ = distilbert_predict_sentence(sentence)
    xlnet_labels, _ = xlnet_predict_sentence(sentence)
    ctbert_labels, _ = ctbert_predict_sentence(sentence)

    # Perform ensemble voting
    ensemble_labels = np.logical_or.reduce([distilbert_labels, xlnet_labels, ctbert_labels], axis=0)

    return ensemble_labels
def ensemble_predict_sentences(sentences):
    # Get individual model predictions for each sentence
    distilbert_labels = np.array([distilbert_predict_sentence(sentence)[0] for sentence in sentences])
    xlnet_labels = np.array([xlnet_predict_sentence(sentence)[0] for sentence in sentences])
    ctbert_labels = np.array([ctbert_predict_sentence(sentence)[0] for sentence in sentences])

    # Perform ensemble voting
    ensemble_labels = np.logical_or.reduce([distilbert_labels, xlnet_labels, ctbert_labels], axis=0)

    return ensemble_labels
# Define weights for each model
distilbert_weight = 0.3
xlnet_weight = 0.4
ctbert_weight = 0.3

# Ensemble function with weighted average
def weighted_ensemble_predict_sentence(sentence):
    # Get individual model predictions
    distilbert_labels, distilbert_confidences = distilbert_predict_sentence(sentence)
    xlnet_labels, xlnet_confidences = xlnet_predict_sentence(sentence)
    ctbert_labels, ctbert_confidences = ctbert_predict_sentence(sentence)

    # Calculate weighted average of model confidences
    weighted_average_confidences = (
        distilbert_weight * distilbert_confidences +
        xlnet_weight * xlnet_confidences +
        ctbert_weight * ctbert_confidences
    ) / (distilbert_weight + xlnet_weight + ctbert_weight)

    # Threshold the weighted average confidences to get the ensemble prediction
    threshold = 0.7
    ensemble_labels = (weighted_average_confidences > threshold).astype(int)

    return ensemble_labels, weighted_average_confidences


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd
import ast 
# data
df = pd.read_csv('final_dataset2.csv',index_col=0)
df['label'] = df['label'].apply(lambda x: ast.literal_eval(x))



label_binarizer = MultiLabelBinarizer()
yt=label_binarizer.fit_transform(df['label'])
# splitting the data

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].values, yt, test_size=0.2, random_state=42
)

# DistilBERT tokenizer and model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert_model_2-20231125T064905Z-001\distilbert_model_2')
# xlnet tokenizer and model
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetForSequenceClassification.from_pretrained("XLNET_model-20231125T122205Z-001\XLNET_model")
# ct-bert 
ctbert_tokenizer = BertTokenizer.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2')
ctbert_model = BertForSequenceClassification.from_pretrained('ctbert_model-20231125T125538Z-001\ctbert_model')

# Function to predict with model

# input_sentence =input()
input_sentence = "As a machine learning student at @IIIT LUCKNOW, I recognize the importance of making informed decisions based on reliable information. It's crucial to consult reputable sources and scientific evidence when assessing the safety and efficacy of vaccines, rather than relying on unfounded concerns or mistrust in vaccine manufacturers."
distilbert_predicted_labels_index, distilbert_confidences = distilbert_predict_sentence(input_sentence)
xlnet_predicted_labels_index, xlnet_confidences = xlnet_predict_sentence(input_sentence)
ctbert_predicted_labels_index, ctbert_confidences = ctbert_predict_sentence(input_sentence)
# print(label_binarizer.classes_[distilbert_predicted_labels_index[0]])
# print(label_binarizer.classes_[xlnet_predicted_labels_index[0]])
# print(label_binarizer.classes_[ctbert_predicted_labels_index[0]])

# print(distilbert_predicted_labels_index, distilbert_confidences )
# print(xlnet_predicted_labels_index, xlnet_confidences)
# Print the result

# print(f"Confidences: {confidences}")
ensemble_labels = ensemble_predict_sentences(val_texts)

# ensemble_labels, ensemble_confidences = weighted_ensemble_predict_sentence(input_sentence)
# ensemble_labels = ensemble_predict_sentence(input_sentence)
predicted_label_names = label_binarizer.classes_[ensemble_labels[0]]
accuracy = accuracy_score(val_labels, predicted_label_names)
print(f"accuracy is {accuracy} ")
