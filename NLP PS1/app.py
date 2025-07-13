# Import libraries
from flask import Flask, request, render_template 
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__) # Creates a Flask app instance 

# Loads the pre-trained Logistic Regression model and TF-IDF vectorizer from .pkl files.
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

@app.route('/') # Home route 
def home():
    return render_template('index.html') #it loads the index.html form for user input.

# Defines the /predict route which accepts POST requests 
@app.route('/predict', methods=['POST'])  
def predict():
    review = request.form['review'] # Retrieves the review text submitted via the form (
    cleaned = preprocess(review)    # Cleans review using the preprocessing function 
    vectorized = tfidf.transform([cleaned]) # Converts the cleaned text into a TF-IDF feature vector.
    prediction = model.predict(vectorized)[0] # Uses the model to predict sentiment.
    result = 'Positive' if prediction == 1 else 'Negative'
    return render_template('index.html', prediction=result) # Shows result 

if __name__ == '__main__': # Starts Flask app only if this file is run directly
    app.run(debug=True)


    