# SMS Spam Classifier - README

## Introduction

This project is an SMS Spam Classifier that uses Natural Language Processing (NLP) and Machine Learning (ML) to classify text messages as spam or not spam. It is built using Python and Streamlit for the web interface.

## Requirements

To run this project, you need the following dependencies:

- Python 3.7+
- Streamlit
- NLTK
- Scikit-learn
- Pickle

Install the required libraries using:

```bash
pip install streamlit nltk scikit-learn
```

Download the NLTK data files:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## How to Run

1. Ensure `vectorizerr.pkl` and `modell.pkl` are in the same directory as `app.py`.
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to the provided URL (usually `http://localhost:8501`).

## Usage

1. Enter the message you want to classify in the text input box.
2. Click the 'Predict' button to classify the message.
3. The app will display whether the message is 'Spam' or 'Not Spam'.

## Code Explanation

### Text Preprocessing Function

```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)
```

### Loading the Model and Vectorizer

```python
with open('vectorizerr.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('modell.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Streamlit App Interface

```python
st.title("SMS Spam Classifier")
input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    if input_sms:
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        prediction = model.predict(vector_input)[0]
        if prediction == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
```

## Error Handling

The app includes error handling to manage missing files and prediction errors, displaying appropriate messages to the user.
