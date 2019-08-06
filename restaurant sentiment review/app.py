from flask import Flask
from flask import render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pickle


app = Flask(__name__)

# LOADING GLOBAL MODEL VARIABLES

model = joblib.load(filename='data/bayesreview.pkl')
processed_reviews = pickle.load(file=open('data/processed_reviews.txt', mode='rb'))

cv = CountVectorizer()
cv.fit(processed_reviews)


# ROUTE PAGES


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        comment = request.form['comment']
        review = [comment]
        review = cv.transform(review).toarray()
        prediction = model.predict(review)
        prediction = prediction[0]

    return render_template('predict.html', prediction=prediction, comment=comment)


if __name__ == '__main__':
    app.run(debug=True)
