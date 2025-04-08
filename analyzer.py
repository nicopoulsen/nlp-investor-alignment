import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

nltk.download('stopwords')

class TextAnalysisFramework:
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

    def wordcount_sankey(self, word_list=None, k=5):
        # We added these stopwords specifically for the sankey
        # This is because after visualizing the sankey with original stopwords we noticed some words that were useless
        # after running it a couple times we removed extra words to make the sankey more meaningful
        CUSTOM_STOPWORDS = set([
            "said", "says", "also", "however", "about", "could", "would", "should", "can",
            "get", "got", "just", "make", "makes", "might", "still", "many", "much", "even",
            "may", "one", "two", "new", "first", "last", "back", "us", "u", "etc", "’s",
            "reuters", "nytimes", "washingtonpost", "cnn", "wsj", "cnbc", "bloomberg", "ft", "economist",
            "theathletic", "front", "bloombergdotorg", "find", "court", "rt", 'breaking', 'monday',
            "financial", "times", "published", "theterminal", "day", "climate", "every", 'heres',
            'global', 'learn', 'mikebloomberg', 'edition', 'page', 'amp', 'writes'
        ])

        def clean_text_keywords(text):
            text = re.sub(r"http\S+", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            words = text.lower().split()
            return [w for w in words if w not in CUSTOM_STOPWORDS]

        labels = []
        sources = []
        targets = []
        values = []
        label_idx = {}

        def get_index(label):
            if label not in label_idx:
                label_idx[label] = len(labels)
                labels.append(label)
            return label_idx[label]

        for label, text_list in self.texts.items():
            word_counts = {}
            for text in text_list:
                words = clean_text_keywords(text)
                for w in words:
                    word_counts[w] = word_counts.get(w, 0) + 1

            if word_list:
                top_words = {w: word_counts[w] for w in word_list if w in word_counts}
            else:
                top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:k])

            for word, count in top_words.items():
                src = get_index(label)
                tgt = get_index(word)
                sources.append(src)
                targets.append(tgt)
                values.append(count)

        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
            link=dict(source=sources, target=targets, value=values)
        ))
        fig.update_layout(title_text="Text to Word Sankey Diagram", font_size=20)
        fig.write_image("wordcount_sankey.pdf")  
        fig.show()

    def subplot_visualization(self):

        sentiments = {
            label: self.data["sentiment"][label]["average"]
            for label in self.texts
        }

        plt.figure(figsize=(10, 5))
        plt.bar(sentiments.keys(), sentiments.values(), color='skyblue')
        plt.ylabel("Average Sentiment")
        plt.title("Average Sentiment per Text")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



