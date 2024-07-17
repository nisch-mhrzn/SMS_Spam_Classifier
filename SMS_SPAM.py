import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Load pre-trained model and vectorizer
try:
    with open('vectorizerr.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('modell.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Streamlit app interface
st.title("SMS Spam Classifier")
input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    if input_sms:
        # Preprocess the input message
        transform_sms = transform_text(input_sms)
        # Vectorize the preprocessed text
        vector_input = tfidf.transform([transform_sms])
        # Predict using the pre-trained model
        try:
            prediction = model.predict(vector_input)[0]
            # Display the result
            if prediction == 1:
                st.header('Spam')
            else:
                st.header('Not Spam')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
