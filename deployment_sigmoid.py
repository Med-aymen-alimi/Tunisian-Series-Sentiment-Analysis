import re
import pickle
import nltk
import streamlit as st

def clean_text(text):
    # Remove non-Arabic characters
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)  # Arabic Unicode range
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove multiple white spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove single characters except spaces
    text = re.sub(r'\b\w\b', '', text)
    # Strip leading and trailing spaces
    text = text.strip()
    return text

def preprocess_text(df):
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    # Keep only alphanumeric and space characters
    df['text'] = df['text'].apply(lambda x: ''.join(char for char in x if char.isnumeric() or char.isalpha() or char.isspace()))
    return df

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

vect = pickle.load(open("vectorizer.pickle", "rb"))
model = pickle.load(open("best_model_sgd.pickle", "rb"))

st.title("Sentiment Analysis sigmoid")
text = st.text_input("Leave your comment")
cleaned_text = clean_text(text)
normalized_text = normalize_arabic(cleaned_text)
X = vect.transform([normalized_text]).toarray()

if st.button("Predict"):
    pred = model.predict(X)
    if pred == 1:
        st.success("Positive")
    elif pred == 0:
        st.info("Neutral")
    else:
        st.error("Negative")
