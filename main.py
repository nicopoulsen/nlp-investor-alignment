import os
import re
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from wordcloud import WordCloud
from textblob import TextBlob

class NLPTweetAnalyzer:
    def __init__(self):
        """
        Initialize the data structures. 
        self.data stores multiple data dictionaries for each label (text source).
          - self.data['texts'][label] = list of text strings (tweets or documents)
          - self.data['word_count'][label] = {word: count} for that label
        """
        self.data = {
            'texts': {},
            'word_count': {},
        }
        self.stopwords = set()
        self.lemmatizer = WordNetLemmatizer()

    def load_stop_words(self, from_file=None):
        """
        Loads NLTK's built-in English stopwords.
        """
        nltk_stop = set(stopwords.words('english'))
        self.stopwords.update(nltk_stop)

    def load_text(self, filename, label=None, parser=None):
        """
        Register a text file (e.g., CSV of tweets) with the library. 
        The label is an optional identifier used in visualizations.

        We'll store:
          - self.data['texts'][label] = list of cleaned text strings
          - self.data['word_count'][label] = {word: count} (excluding stopwords)
        """
        if label is None:
            label = os.path.basename(filename)  # fallback label from filename

        # Assumes tweets.csv with a column named 'Text'
        df = pd.read_csv(filename)
        if 'Text' not in df.columns:
            raise ValueError(f"File {filename} must have a 'Text' column!")

        texts = df['Text'].astype(str).tolist()

        if parser is not None:
            cleaned_texts = [parser(t) for t in texts]
        else:
            cleaned_texts = [self.default_parser(t) for t in texts]

        # Store the cleaned texts
        self.data['texts'][label] = cleaned_texts

        # Count words (excluding stopwords)
        word_counter = Counter()
        for text in cleaned_texts:
            words = text.split()
            words = [w for w in words if w and w not in self.stopwords]
            word_counter.update(words)

        self.data['word_count'][label] = dict(word_counter)

    def get_wordnet_pos(self, treebank_tag):
        """
        Map 'pos_tag' output (e.g., 'NN', 'VB') to WordNetLemmatizer's expected
        part-of-speech format (wordnet.NOUN, wordnet.VERB, etc.).
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # default to noun

    def default_parser(self, text):
        """
        A parser that:
          1. Lowercases
          2. Removes punctuation (#, !, ?, ., etc.)
          3. Tokenizes
          4. POS-tags
          5. Lemmatizes
          6. Joins back into a string
        Customize to handle hashtags, mentions, etc. as needed.
        """
        # Lowercase & basic punctuation removal
        text = text.lower()
        text = re.sub(r'[\\!\\?\"\\.,;:@#]', '', text)
        text = text.strip()

        # Tokenize
        tokens = word_tokenize(text)

        # POS tag each token
        tagged_tokens = pos_tag(tokens)

        # Lemmatize each token based on its POS
        lemma_tokens = [
            self.lemmatizer.lemmatize(token, self.get_wordnet_pos(pos))
            for token, pos in tagged_tokens
        ]

        # Return a single string
        return ' '.join(lemma_tokens)

    def wordcount_sankey(self, word_list=None, k=5):
        """
        A Sankey diagram mapping each label (text source) to words.
        Thickness represents the word count in that text.

        """
        # 1. Gather words
        if not word_list:
            all_top_words = []
            for label, wc_dict in self.data['word_count'].items():
                top_k = sorted(wc_dict.items(), key=lambda x: x[1], reverse=True)[:k]
                all_top_words.extend([word for word, _ in top_k])
            word_list = list(set(all_top_words))

        labels = list(self.data['word_count'].keys())
        label_count = len(labels)

        # Node labels = text labels + words
        node_labels = labels + word_list

        word_index_map = {}
        for i, w in enumerate(word_list):
            word_index_map[w] = label_count + i

        source = []
        target = []
        value = []

        # Build link data
        for i, lbl in enumerate(labels):
            wc_dict = self.data['word_count'][lbl]
            for w in word_list:
                count_val = wc_dict.get(w, 0)
                if count_val > 0:
                    source.append(i)
                    target.append(word_index_map[w])
                    value.append(count_val)

        # Build Sankey with Plotly
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=node_labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        fig.update_layout(title_text='Text-to-Word Sankey Diagram', font_size=12)
        fig.show()

    def second_visualization(self, misc_parameters=None):
        """
        Visualization with subplots, e.g. top 10 words as bar charts, 
        one subplot per text label. 
        """
        labels = list(self.data['word_count'].keys())
        num_labels = len(labels)

        fig, axes = plt.subplots(nrows=num_labels, ncols=1, figsize=(8, 4 * num_labels))
        if num_labels == 1:
            axes = [axes]  # so we can iterate

        for ax, label in zip(axes, labels):
            wc_dict = self.data['word_count'][label]
            top_10 = sorted(wc_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            words, counts = zip(*top_10) if top_10 else ([], [])
            ax.bar(words, counts)
            ax.set_title(f'Top Words - {label}')
            ax.set_ylabel('Count')
            ax.set_xticklabels(words, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def third_visualization(self, misc_parameters=None):
        """
        A comparative overlay of data from each text label. 
        Here: we'll plot random data for each label over 10 time points.
        Replace with actual time series data from tweets or another metric.
        """
        labels = list(self.data['word_count'].keys())
        x = np.arange(10)  # e.g. 10 time points

        plt.figure(figsize=(10, 6))
        for label in labels:
            y = np.random.rand(10)  # placeholder
            plt.plot(x, y, marker='o', label=label)

        plt.title('Example Comparative Overlay')
        plt.xlabel('Time Index (example)')
        plt.ylabel('Some Metric')
        plt.legend()
        plt.show()
