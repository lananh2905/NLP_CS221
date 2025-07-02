import streamlit as st
import pandas as pd
import os
import plotly.graph_objs as go
from utils.preprocessing import readData
from utils.preprocessing import remove_punctuation, remove_special_characters, remove_html_tags
from keras.models import load_model
from utils.classify import feedbackSentimentAnalysis
import numpy as np
from copy import deepcopy
import pickle
from underthesea import word_tokenize

# Đặt cấu hình trang
st.set_page_config(layout="wide")

# Load Model
model = load_model("models/best_lstm_model.h5")

with open("utils/words_dict.pkl", "rb") as file:
    words = pickle.load(file)

DESIRED_SEQUENCE_LENGTH = 205

def tokenize_vietnamese_sentence(sentence):
    return word_tokenize(remove_special_characters(remove_html_tags(remove_punctuation(sentence.lower()))))

def sent2vec(message, word_dict=words):
    tokens = tokenize_vietnamese_sentence(message)
    vectors = [word_dict[token] for token in tokens if token in word_dict]
    return np.array(vectors, dtype=float) if vectors else np.zeros((1, 200))

def X_to_vectors(sentences):
    return [sent2vec(sentence) for sentence in sentences]

def pad_sequences(X, desired_sequence_length=205):
    X_copy = deepcopy(X)
    for i, x in enumerate(X):
        if x.shape[0] < desired_sequence_length:
            pad = np.zeros((desired_sequence_length - x.shape[0], 200))
            X_copy[i] = np.concatenate((x, pad))
    return np.array(X_copy, dtype=float)

def predictions(sentences):
    vectors = X_to_vectors(sentences)
    sequences = pad_sequences(vectors)
    predictions = model.predict(sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    sentiments = [feedbackSentimentAnalysis(label) for label in predicted_labels]
    return pd.DataFrame({"feedback": sentences, "sentiment": sentiments})

def renderPage():
    st.title("Corpus Analysis")

    # File Upload chỉ được gọi một lần
    uploaded_file = st.file_uploader("Browse Corpus", type=["csv", "txt"], key="file_uploader_1")

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            # Read CSV file with proper handling of line breaks
            df = pd.read_csv(uploaded_file, skip_blank_lines=True)

            # Rename the last column to "sentiment"
            df.rename(columns={df.columns[-1]: "sentiment"}, inplace=True)

            # Extract each review from the last column, ensuring each line is separate
            sentences = df["sentiment"].dropna().tolist()

        elif file_extension == 'txt':
            # Read and split the content of the TXT file line by line
            content = uploaded_file.read().decode("utf-8")
            sentences = content.splitlines()

        # Remove empty strings or whitespace-only entries
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Use the sentences for sentiment analysis
        df = predictions(sentences)

        # Display DataFrame
        st.subheader("Predictions:")
        st.dataframe(df, width=800)

        # Generate Pie Chart for sentiment distribution
        labels = df["sentiment"].value_counts().index
        values = df["sentiment"].value_counts().values
        trace = go.Pie(labels=labels, values=values, name="sentiment")

        # Create Pie Chart figure
        fig = go.Figure(data=[trace])
        fig.update_layout(title="Sentiment Distribution", showlegend=True)
        st.plotly_chart(fig)

