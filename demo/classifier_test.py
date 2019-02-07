from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred = clf.get_prediction_message("i really liked the movie and had fun")

print(pred)

pred = clf.get_prediction_message("this movie was terrible and bad")

print(pred)