import streamlit as st
import joblib
import gzip
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

#function to load pickle files
@st.cache_resource
def load_resources():

    model_path = os.path.abspath("model/MNB_model_v2.pkl")
    vectorizer_path = os.path.abspath("model/vectorizer_v2.pkl")

    st.write(f"🔍 Checking files:")
    st.write(f"📂 Model path: {model_path} → Exists: {os.path.exists(model_path)}")
    st.write(f"📂 Vectorizer path: {vectorizer_path} → Exists: {os.path.exists(vectorizer_path)}")

    if not os.path.exists(model_path):
        st.error("🚨 Model file is missing!")
        return None, None

    if not os.path.exists(vectorizer_path):
        st.error("🚨 Vectorizer file is missing!")
        return None, None

    try:
        MNB_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        st.success("✅ Model and vectorizer loaded successfully!")
    except Exception as e:
        st.error(f"🚨 Error loading model/vectorizer: {e}")
        return None, None

    return MNB_model, vectorizer

 

#load model and vectorizer 
MNB_model , vectorizer = load_resources()


#function to predict a single text
def predict_text_MNB(text):
    if vectorizer is None:
        st.error("Vectorizer is not loaded!")
        return ["Unknown"]

    if MNB_model is None:
        st.error("Model is not loaded!")
        return ["Unknown"]
    
    text = vectorizer.transform([text])
    prediction = MNB_model.predict(text)
    return prediction

def confusion(predicted_y, y):
  fig, ax = plt.subplots(figsize=(12,8))
  languages = np.unique(y)
  cm2 = confusion_matrix(y, predicted_y, labels= languages)
  sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax)
  ax.set_xticks(np.arange(len(languages)) + 0.5, languages, rotation= 45)
  ax.set_yticks(np.arange(len(languages)) + 0.5, languages, rotation = 360)
  plt.tight_layout()
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  st.pyplot(fig)

def predict_dataset(file):
   dataset = pd.read_csv(file, delimiter= ',')
   x_data = dataset['text']
   y_data = dataset['language']
   x_data_vectorized = vectorizer.transform(x_data)
   data_pred = MNB_model.predict(x_data_vectorized)
   df = pd.concat([x_data, y_data, pd.Series(data_pred)], axis=1)
   df.columns = ['Text', 'Actual', 'Predicted']
   accuracy = accuracy_score(y_data, data_pred)
   dataframe = st.table(df)
   conf = confusion(data_pred, y_data)
   return  conf, dataframe


st.header('LANGUAGE DETECTION')
st.divider()
tab1, tab2 = st.tabs(['Single Text ', 'Dataset'])
with tab1:
 st.subheader('Detect a new text')
 text = st.text_input(label= 'ENTER TEXT:')

 if text:
  prediction = predict_text_MNB(text)
  st.write(f" Detected language: {prediction[0]}")

st.divider()

with tab2:
 st.subheader('Detect a new dataset')
 st.caption('NOTE: Dataset should have two columns (text and language)')
 file = st.file_uploader('UPLOAD CSV/TXT FILE:')
 if file: 
  pred = predict_dataset(file)

