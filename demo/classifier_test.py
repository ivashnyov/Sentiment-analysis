from sentiment_classifier import SentimentClassifier

clf = SentimentClassifier()

pred = clf.get_prediction_message("Это отличный банк, просто чудесный")

print pred