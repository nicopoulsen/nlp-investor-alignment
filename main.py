from analyzer import TweetAnalyzer

def main():
    analyzer = TweetAnalyzer()

    analyzer.load_text('data/nytimes.csv', label='NYTimes')
    analyzer.load_text('data/Reuters.csv', label='Reuters')
    analyzer.load_text('data/SeekingAlpha.csv', label='SeekingAlpha')
    analyzer.load_text('data/YahooFinance.csv', label='YahooFinance')
    analyzer.load_text('data/washingtonpost.csv', label='WashingtonPost')
    analyzer.load_text('data/CNN.csv', label='CNN')
    analyzer.load_text('data/WSJ.csv', label='wsj')
    analyzer.load_text('data/FinancialTimes.csv', label='FT')
    analyzer.load_text('data/CNBC.csv', label='cnbc')
    analyzer.load_text('data/Bloomberg.csv', label='bloomberg')
    analyzer.load_text('data/TheEconomist.csv', label='economist')

    summaries = analyzer.get_summary()
    for source, summary in summaries.items():
        print(f"\n--- {source} ---")
        print("Avg Sentiment:", summary["Avg sentiment"])
        print("Top Words:", summary["Top Words"])

    print("\n\n Sentiment Analysis:Top 3 topics per outlet:")
    topic_sentiment = analyzer.analyze_top_topics_sentiment()
    for outlet, topics in topic_sentiment.items():
        print(f"\n== {outlet} ==")
        for topic, sentiment in topics.items():
            print(f"  {topic}: {sentiment}")
## probably worth checking if theres a way to 
    macro_topics = ["inflation", "interest rates", "recession", "sp500", "jobs", "unemployment", "trump", "tariff", "tariffs"]


    print("\n\nSentiment on Defined Macroeconomic Topics:")
    macro_results = analyzer.analyze_defined_topics_sentiment(macro_topics)
    for outlet, topics in macro_results.items():
        print(f"\n== {outlet} ==")
        for topic, score in topics.items():
            print(f"{topic}: {score}")

if __name__ == "__main__":
    main()
