import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pickle

# Load the saved pickle model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Description
description = """
# Personality Detection App

Unleash the Power of Personality Analysis with the Myers Briggs Type Indicator (MBTI)!

The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axes:

- *Introversion (I) – Extroversion (E)*
- *Intuition (N) – Sensing (S)*
- *Thinking (T) – Feeling (F)*
- *Judging (J) – Perceiving (P)*

Gain insights into yourself and others by exploring the fascinating world of personality types!

---

Built with ❤ by [Sanjana singamsetty](https://github.com/sanjana-singamsetty)
"""

# Render the description using st.markdown
st.markdown(description)

# Function to clean text
def clear_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(data):
        sentence = sentence.lower()
        # Removing links from text data
        sentence = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)
        # Removing other symbols
        sentence = re.sub(r'[^0-9a-z]', ' ', sentence)
        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text, data_length

# Streamlit app
# Input box for user input
user_input = st.text_area("Enter your text here:", "")

# Cleaning the input text
if st.button("Process"):
    # Check if user has entered any input
    if not user_input:
        st.error("Please enter some text!")
    else:
        # Cleaning the input
        cleaned_input, _ = clear_text([user_input])
        # Vectorizing the input using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', tokenizer=WordNetLemmatizer().lemmatize)
        # Fit the vectorizer on cleaned input text
        vectorizer.fit(cleaned_input)
        # Transforming the input text into TF-IDF matrix
        tfidf_matrix = vectorizer.transform(cleaned_input)

        # Check if the number of features is less than 5000
        if tfidf_matrix.shape[1] < 5000:
            # Pad the TF-IDF matrix with zeros to have 5000 features
            tfidf_matrix = np.pad(tfidf_matrix.toarray(), ((0, 0), (0, 5000 - tfidf_matrix.shape[1])), mode='constant')

        # Making predictions using the loaded model
        prediction = model.predict(tfidf_matrix)
        # Displaying the predicted result
        st.success(f"Predicted Personality Type: {prediction[0]}")
data = {
    "Personality Type": ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", 
                         "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"],
    "Number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Render the table using st.table
st.table(df)