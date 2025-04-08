import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import nltk

nltk.download('stopwords')

class TweetAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.data = defaultdict(dict)
        self.texts = {}

    def load_stop_words(self, stopfile=None):
        if stopfile:
            with open(stopfile, 'r') as f:
                self.stop_words.update(word.strip() for word in f.readlines())

    def clean_text(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.lower()
        words = text.split()
        return [word for word in words if word not in self.stop_words]

    def load_text(self, filename, label=None, parser=None):
        label = label or filename
        df = pd.read_csv(filename)
        if 'text' not in df.columns:
            raise ValueError("CSV must have a 'text' column")
        texts = df['text'].dropna().tolist()

        cleaned_texts = []
        word_counts = defaultdict(int)
        sentiment_scores = []

        for text in texts:
            if parser:
                text = parser(text)
            words = self.clean_text(text)
            for word in words:
                word_counts[word] += 1
            cleaned_texts.append(" ".join(words))

            sentiment = TextBlob(text).sentiment.polarity
            sentiment_scores.append(sentiment)

        self.texts[label] = cleaned_texts
        self.data['word_count'][label] = dict(word_counts)
        self.data['sentiment'][label] = {
            'average': sum(sentiment_scores) / len(sentiment_scores),
            'individual': sentiment_scores
        }

    def get_summary(self):
        return {
            label: {
                "Avg Sentiment": round(self.data["sentiment"][label]["average"], 3),
                "Top Words": sorted(self.data["word_count"][label].items(), key=lambda x: x[1], reverse=True)[:5]
            }
            for label in self.texts
        }
    

    def analyze_defined_topics_sentiment(self, topic_keywords):
        topic_sentiment = {}

        for label, texts in self.texts.items():
            topic_sentiment[label] = {}
            for topic in topic_keywords:
                topic = topic.lower()
                matching_sentiments = []
                for i, doc in enumerate(texts):
                    if topic in doc:
                        raw_text = doc  
                        matching_sentiments.append(TextBlob(raw_text).sentiment.polarity)
                if matching_sentiments:
                    topic_sentiment[label][topic] = round(sum(matching_sentiments) / len(matching_sentiments), 3)
                else:
                    topic_sentiment[label][topic] = None  

        return topic_sentiment


    def analyze_top_topics_sentiment(self, top_k=3):
        topic_sentiment = {}

        for label, texts in self.texts.items():
            word_counts = self.data["word_count"][label]
            top_topics = [word for word, _ in Counter(word_counts).most_common(top_k)]

            topic_sentiment[label] = {}

            for topic in top_topics:
                matching_sentiments = []
                for i, doc in enumerate(texts):
                    if topic in doc:
                        raw_text = doc  
                        matching_sentiments.append(TextBlob(raw_text).sentiment.polarity)
                if matching_sentiments:
                    topic_sentiment[label][topic] = round(sum(matching_sentiments) / len(matching_sentiments), 3)
                else:
                    topic_sentiment[label][topic] = None

        return topic_sentiment
