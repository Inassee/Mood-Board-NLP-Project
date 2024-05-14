from textblob import TextBlob

def process_text(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    else:
        return 'negative'
