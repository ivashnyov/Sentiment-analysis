from sentiment_classifier import SentimentClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

classifier = SentimentClassifier()


@app.route("/", methods=["POST", "GET"])
def index_page(text="", prediction_message="", tonality=""):
    if request.method == "POST":
        text = request.form["text"]
        prediction_message = classifier.get_prediction_message(text)
        print(prediction_message)
        tonality = classifier.get_tonality(prediction_message)
    return render_template('sentiment.html', text=text, prediction_message=prediction_message, tonality=tonality)


if __name__ == "__main__":
    app.run()
