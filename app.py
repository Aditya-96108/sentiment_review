from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        vectorized = vectorizer.transform([review])
        prediction = model.predict(vectorized)[0]
        result = "Positive" if prediction == 1 else "Negative"
        return render_template('index.html', result=result, review=review)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
