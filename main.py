from analyzer import TextAnalysisFramework

def main():
    analyzer = TextAnalysisFramework()

    analyzer.load_text('data/nytimes.csv', label='NYTimes')
    analyzer.load_text('data/Reuters.csv', label='Reuters')
    analyzer.load_text('data/SeekingAlpha.csv', label='SeekingAlpha')
    analyzer.load_text('data/YahooFinance.csv', label='YahooFinance')
    analyzer.load_text('data/washingtonpost.csv', label='WashingtonPost')
    analyzer.load_text('data/CNN.csv', label='CNN')
    analyzer.load_text('data/WSJ.csv', label='WSJ')
    analyzer.load_text('data/FinancialTimes.csv', label='FT')
    analyzer.load_text('data/CNBC.csv', label='CNBC')
    analyzer.load_text('data/Bloomberg.csv', label='Bloomberg')
    analyzer.load_text('data/TheEconomist.csv', label='Economist')

    summaries = analyzer.get_summary()
    for source, summary in summaries.items():
        print(f"\n--- {source} ---")
        print("Avg Sentiment:", summary["Avg Sentiment"])
        print("Top Words:", summary["Top Words"])

    print("\n\n🔍 Sentiment Analysis: Top 3 Topics per Outlet")
    topic_sentiment = analyzer.analyze_top_topics_sentiment()
    for outlet, topics in topic_sentiment.items():
        print(f"\n== {outlet} ==")
        for topic, sentiment in topics.items():
            print(f"  {topic}: {sentiment}")
    macro_topics = [
        "inflation", "interest rates", "recession", "sp500", "jobs", "unemployment", "economy",
        "growth", "federal reserve", "monetary policy", "tariffs", "trade", "gdp", "housing",
        "trump", "deficit", "debt ceiling", "consumer spending", "yields", "banking"
    ]
    print("\n\nSentiment on Defined Macroeconomic Topics:")
    macro_results = analyzer.analyze_defined_topics_sentiment(macro_topics)
    for outlet, topics in macro_results.items():
        print(f"\n== {outlet} ==")
        for topic, score in topics.items():
            print(f"{topic}: {score}")

    analyzer.wordcount_sankey(k=5)
    analyzer.subplot_visualization()
    analyzer.plot_top_words_subplots(top_k=3)

if __name__ == "__main__":
    main()
