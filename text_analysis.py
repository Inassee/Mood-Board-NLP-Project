from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

def process_text(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    sentiment = sentiment_scores['compound']
    print(f"Sentiment score: {sentiment}, Scores: {sentiment_scores}")
    if sentiment >= 0.05:
        return 'positive'
    elif sentiment <= -0.05:
        return 'negative'
    else:
        return 'neutral'
