import pypickle as py
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
url="https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset"
st.link_button("Go to Data source",url)
st.title("X (fka TWITTER) COMMENT REVIEW")
st.header("Review difference comments and determine if it is Negative, Neutral or Positive")
model=py.load("lg.pkl")
vectorizer=py.load("vector.pkl")
user_input=st.text_input('Enter Comment')
predictions=vectorizer.transform([user_input])
predictx=model.predict(predictions)
if st.button("ENTER"):
  if predictx==1:
    st.write("Postive")
  elif predictx==0:
    st.write("Neutral")
  else:
    st.write("Negative")
st.markdown("""<div style='text-align: center;'><p style='font-size: small;'>this app used machine learning to train itself with the dataset and accuracy is 86%</p></div>""", unsafe_allow_html=True)
st.markdown("""<div style='text-align: center;'><p style='font-size: small;'>by Omatsola</p></div>""", unsafe_allow_html=True)