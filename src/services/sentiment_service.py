import tweepy
from textblob import TextBlob
import os
from dotenv import load_dotenv
import praw # Added for Reddit integration
import requests # Added for EODHD news API

load_dotenv() # Load environment variables from .env file

class SentimentDataService:
    def __init__(self):
        # Initialize Twitter client
        # It's recommended to use environment variables for API keys
        self.consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
        self.consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN") # For Twitter API v2

        if self.bearer_token:
            self.twitter_client_v2 = tweepy.Client(bearer_token=self.bearer_token)
            self.twitter_api_v1 = None
        elif self.consumer_key and self.consumer_secret and self.access_token and self.access_token_secret:
            auth = tweepy.OAuth1UserHandler(
                self.consumer_key, self.consumer_secret,
                self.access_token, self.access_token_secret
            )
            self.twitter_api_v1 = tweepy.API(auth)
            self.twitter_client_v2 = None
        else:
            print("Warning: Twitter API credentials not found. Sentiment analysis from Twitter will not be available.")
            self.twitter_client_v2 = None
            self.twitter_api_v1 = None

        # Initialize Reddit client
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

        if self.reddit_client_id and self.reddit_client_secret and self.reddit_user_agent:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                    check_for_async=False # Added to avoid Prawcoreasyncwarning if not using async
                )
                # Test connection
                print(f"Reddit client initialized for user: {self.reddit_client.user.me()}")
            except Exception as e:
                print(f"Error initializing Reddit client: {e}")
                self.reddit_client = None
        else:
            print("Warning: Reddit API credentials not found. Sentiment analysis from Reddit will not be available.")
            self.reddit_client = None

        # Initialize EODHD API token
        self.eodhd_api_token = os.getenv("EODHD_API_TOKEN")
        if not self.eodhd_api_token:
            print("Warning: EODHD_API_TOKEN not found. Sentiment analysis from News (EODHD) will not be available.")

    def fetch_social_sentiment(self, symbol: str, timeframe: str, source: str = "twitter", limit: int = 10):
        """
        Fetches social sentiment for a given symbol and timeframe from a specified source.
        Supported sources: 'twitter', 'reddit', 'news'.
        Timeframe parameter is not yet fully implemented (fetches recent data).
        Limit parameter controls the number of items fetched (tweets or posts).
        """
        if source.lower() == "twitter":
            if not self.twitter_client_v2 and not self.twitter_api_v1:
                print("Twitter client not initialized. Cannot fetch Twitter sentiment.")
                return {"symbol": symbol, "timeframe": timeframe, "source": source, "error": "Twitter client not initialized"}

            texts_to_analyze = []
            query = f"{symbol} -is:retweet lang:en" 

            try:
                if self.twitter_client_v2:
                    response = self.twitter_client_v2.search_recent_tweets(query=query, max_results=limit, tweet_fields=['text'])
                    if response.data:
                        for tweet in response.data:
                            texts_to_analyze.append(tweet.text)
                elif self.twitter_api_v1:
                    fetched_tweets = self.twitter_api_v1.search_tweets(q=query, count=limit, lang='en')
                    for tweet in fetched_tweets:
                        texts_to_analyze.append(tweet.text)
                
                if not texts_to_analyze:
                    return self._format_sentiment_results(symbol, timeframe, source, [], "no_content_found")

                return self._analyze_texts_with_textblob(symbol, timeframe, source, texts_to_analyze)

            except Exception as e:
                print(f"Error fetching/analyzing Twitter sentiment for {symbol}: {e}")
                return self._format_sentiment_results(symbol, timeframe, source, [], f"twitter_api_error: {str(e)}")
        
        elif source.lower() == "reddit":
            if not self.reddit_client:
                print("Reddit client not initialized. Cannot fetch Reddit sentiment.")
                return {"symbol": symbol, "timeframe": timeframe, "source": source, "error": "Reddit client not initialized"}
            
            texts_to_analyze = []
            # Try to find a relevant subreddit. This is a simple approach.
            # For financial symbols, common subreddits might be r/CryptoCurrency, r/stocks, r/wallstreetbets, or symbol-specific ones like r/bitcoin
            # A more robust solution would involve subreddit discovery or allowing user to specify.
            subreddit_name = symbol.lower() # Try symbol as subreddit name first
            if symbol.upper() in ["BTC", "BITCOIN"] : subreddit_name = "bitcoin"
            if symbol.upper() in ["ETH", "ETHEREUM"] : subreddit_name = "ethereum"
            # Add more mappings or a more generic search/discovery mechanism

            try:
                print(f"Attempting to fetch from subreddit: r/{subreddit_name} for symbol {symbol}")
                subreddit = self.reddit_client.subreddit(subreddit_name)
                # Fetching hot posts. Could also use .new() or .top(time_filter=timeframe) if timeframe is mapped e.g. "day", "week"
                # PRAW handles pagination implicitly for iterators like .hot(), .new(), .top()
                # The `limit` parameter for these methods controls how many items PRAW will retrieve in total.
                posts = subreddit.hot(limit=limit)
                
                for post in posts:
                    texts_to_analyze.append(post.title) # Analyze title
                    if post.selftext: # Analyze body if it exists (not a link post)
                        texts_to_analyze.append(post.selftext)
                    # Optionally, could also fetch and analyze comments: post.comments.list()
                
                if not texts_to_analyze:
                     return self._format_sentiment_results(symbol, timeframe, source, [], "no_content_found_in_subreddit")

                return self._analyze_texts_with_textblob(symbol, timeframe, source, texts_to_analyze)

            except praw.exceptions.PRAWException as e: # More specific PRAW errors
                print(f"PRAW API error fetching Reddit sentiment for {symbol} from r/{subreddit_name}: {e}")
                # Try a more general subreddit if specific one fails or is not found, e.g. r/CryptoCurrency for crypto symbols
                if "crypto" in symbol.lower() or symbol.upper() in ["BTC", "ETH"]: # Basic check
                    try:
                        print(f"Retrying with r/CryptoCurrency for symbol {symbol}")
                        subreddit = self.reddit_client.subreddit("CryptoCurrency")
                        # Search within the general subreddit for the symbol
                        posts = subreddit.search(query=symbol, sort="hot", limit=limit) # or sort='relevance', 'new'
                        for post in posts:
                            texts_to_analyze.append(post.title)
                            if post.selftext:
                                texts_to_analyze.append(post.selftext)
                        if not texts_to_analyze:
                            return self._format_sentiment_results(symbol, timeframe, source, [], f"no_content_found_in_r/CryptoCurrency_after_retry: {str(e)}")
                        return self._analyze_texts_with_textblob(symbol, timeframe, source, texts_to_analyze)
                    except Exception as e_retry:
                        print(f"Error fetching/analyzing Reddit sentiment from r/CryptoCurrency for {symbol}: {e_retry}")
                        return self._format_sentiment_results(symbol, timeframe, source, [], f"reddit_api_error_retry: {str(e_retry)}")
                else:
                    return self._format_sentiment_results(symbol, timeframe, source, [], f"reddit_api_error: {str(e)}")        
            except Exception as e:
                print(f"Generic error fetching/analyzing Reddit sentiment for {symbol}: {e}")
                return self._format_sentiment_results(symbol, timeframe, source, [], f"reddit_generic_error: {str(e)}")

        elif source.lower() == "news":
            if not self.eodhd_api_token:
                return self._format_sentiment_results(symbol, timeframe, source, [], "eodhd_api_token_not_found")
            
            # EODHD uses .US suffix for US stocks, .CC for crypto. Need to map common symbols.
            eodhd_symbol = symbol.upper()
            if eodhd_symbol in ["BTC", "ETH"]: # Add other cryptos as needed
                eodhd_symbol += "-USD.CC" # Example: BTC-USD.CC, ETH-USD.CC
            elif not ( "." in eodhd_symbol or "-" in eodhd_symbol): # Basic check if it might be a US stock
                eodhd_symbol += ".US" # Assuming US stock if no exchange/suffix, e.g., AAPL.US
            
            # Timeframe mapping: EODHD uses `from` and `to` dates. 
            # For simplicity, if timeframe is like "1d", fetch news for the last day.
            # A more robust mapping from our timeframe (e.g., "1h", "4h", "1d") to date ranges is needed.
            # For now, let's fetch recent news (EODHD default or up to limit).
            # The `limit` parameter for EODHD news is up to 1000, default 50.
            news_api_url = f"https://eodhd.com/api/news?s={eodhd_symbol}&limit={limit}&api_token={self.eodhd_api_token}&fmt=json"
            # Can add &from=YYYY-MM-DD and &to=YYYY-MM-DD if timeframe is processed

            try:
                print(f"Fetching news from EODHD for {eodhd_symbol}...")
                response = requests.get(news_api_url)
                response.raise_for_status() # Raise an exception for HTTP errors
                articles = response.json()

                if not articles or not isinstance(articles, list):
                    return self._format_sentiment_results(symbol, timeframe, source, [], f"no_articles_found_or_bad_format_for_{eodhd_symbol}")

                # Use EODHD's provided sentiment if available and seems reasonable, otherwise use TextBlob
                # EODHD sentiment object example: {"polarity": 0.23, "neg": 0.1, "neu": 0.7, "pos": 0.2}
                # For consistency, we might prefer to always use TextBlob for now.
                # Let's try to use TextBlob on title + content snippet for now to compare with Twitter/Reddit
                texts_to_analyze = []
                for article in articles:
                    text_content = article.get('title', '')
                    if article.get('content'):
                        # Using a snippet of content to avoid very long texts for TextBlob
                        text_content += ". " + article.get('content')[:500] 
                    if text_content.strip():
                         texts_to_analyze.append(text_content.strip())
                
                if not texts_to_analyze:
                    return self._format_sentiment_results(symbol, timeframe, source, [], f"no_text_content_extracted_for_{eodhd_symbol}")

                return self._analyze_texts_with_textblob(symbol, timeframe, source, texts_to_analyze)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching news from EODHD for {eodhd_symbol}: {e}")
                return self._format_sentiment_results(symbol, timeframe, source, [], f"eodhd_api_request_error: {str(e)}")
            except Exception as e:
                print(f"Generic error processing EODHD news for {eodhd_symbol}: {e}")
                return self._format_sentiment_results(symbol, timeframe, source, [], f"eodhd_generic_error: {str(e)}")

        else:
            print(f"Source '{source}' not supported yet.")
            return self._format_sentiment_results(symbol, timeframe, source, [], f"Source '{source}' not supported")

    def _analyze_texts_with_textblob(self, symbol, timeframe, source, texts_to_analyze):
        sentiments_analyzed_details = []
        total_polarity = 0
        total_subjectivity = 0

        for text in texts_to_analyze:
            analysis = TextBlob(text)
            sentiments_analyzed_details.append({
                "text_snippet": text[:100] + "..." if len(text) > 100 else text, # Store a snippet
                "polarity": analysis.sentiment.polarity,
                "subjectivity": analysis.sentiment.subjectivity
            })
            total_polarity += analysis.sentiment.polarity
            total_subjectivity += analysis.sentiment.subjectivity
        
        avg_polarity = total_polarity / len(texts_to_analyze) if texts_to_analyze else 0
        avg_subjectivity = total_subjectivity / len(texts_to_analyze) if texts_to_analyze else 0
        sentiment_score = avg_polarity # Simplified score

        return {
            "symbol": symbol, 
            "timeframe": timeframe, 
            "source": source,
            "sentiment_score": sentiment_score,
            "avg_polarity": avg_polarity,
            "avg_subjectivity": avg_subjectivity,
            "content_items_analyzed_count": len(texts_to_analyze),
            # "analyzed_item_details": sentiments_analyzed_details # Optionally return individual analysis snippets
            "status": "success"
        }

    def _format_sentiment_results(self, symbol, timeframe, source, analyzed_items_details, status_message):
         return {
            "symbol": symbol, 
            "timeframe": timeframe, 
            "source": source,
            "sentiment_score": 0,
            "avg_polarity": 0,
            "avg_subjectivity": 0,
            "content_items_analyzed_count": 0,
            "analyzed_item_details": analyzed_items_details,
            "status": status_message 
        }

    def calculate_sentiment_momentum(self, sentiment_data_series):
        # Placeholder for calculating sentiment momentum
        # This would involve analyzing a time series of sentiment data
        print("Calculating sentiment momentum...")
        # In a real implementation, this would return momentum indicators
        return {"momentum": "positive_increasing", "change_rate": 0.1}

    def integrate_sentiment_signals(self, technical_analysis_data, sentiment_data):
        # Placeholder for integrating sentiment with technical analysis
        print("Integrating sentiment signals with technical analysis...")
        # This would combine insights from both analyses
        combined_analysis = {
            "technical": technical_analysis_data,
            "sentiment": sentiment_data,
            "overall_outlook": "bullish_strengthened_by_sentiment"
        }
        return combined_analysis

    def get_educational_content(self):
        # Placeholder for educational content
        content = {
            "title": "Understanding Sentiment Analysis in Trading",
            "introduction": "Sentiment analysis gauges the overall emotional tone or attitude of market participants towards a specific asset or the market as a whole.",
            "key_concepts": [
                "Definition: Measuring collective mood (bullish, bearish, neutral).",
                "Data Sources: Social media (Twitter, Reddit), news articles, financial blogs.",
                "Sentiment Indicators: Bull/Bear Ratio, Fear & Greed Index, social media sentiment scores.",
                "Application: Contrarian indicator (extreme sentiment often precedes reversals) or confirmation tool.",
                "Limitations: Can be noisy, susceptible to manipulation, requires careful interpretation."
            ],
            "market_psychology_link": "Sentiment analysis is deeply rooted in market psychology, reflecting crowd behavior, biases (like herding), and emotional decision-making.",
            "integration_with_ta": "Sentiment can confirm or contradict technical signals. For example, strong bullish sentiment during an overbought condition (identified by TA) might signal an impending pullback."
        }
        return content

# Example Usage (for testing purposes, will be removed or integrated into CLI/main app)
if __name__ == '__main__':
    sentiment_service = SentimentDataService()

    # Ensure you have a .env file with your API credentials:
    # TWITTER_BEARER_TOKEN=your_bearer_token
    # REDDIT_CLIENT_ID=your_client_id
    # REDDIT_CLIENT_SECRET=your_client_secret
    # REDDIT_USER_AGENT=your_user_agent (e.g., my_sentiment_app/0.1 by your_username)
    # EODHD_API_TOKEN=your_eodhd_api_token

    # Fetch sentiment from Twitter
    btc_symbol = "BTC"
    twitter_sentiment_btc = sentiment_service.fetch_social_sentiment(btc_symbol, "1d", source="twitter", limit=3)
    print(f"Fetched Twitter Sentiment for {btc_symbol}: {twitter_sentiment_btc}")

    # Fetch sentiment from Reddit
    eth_symbol = "ETH"
    reddit_sentiment_eth = sentiment_service.fetch_social_sentiment(eth_symbol, "1d", source="reddit", limit=3) # Fetch 3 items (posts)
    print(f"Fetched Reddit Sentiment for {eth_symbol}: {reddit_sentiment_eth}")
    
    # Fetch sentiment from News (EODHD)
    aapl_symbol = "AAPL" # Example stock
    news_sentiment_aapl = sentiment_service.fetch_social_sentiment(aapl_symbol, "1d", source="news", limit=3)
    print(f"Fetched News Sentiment for {aapl_symbol} (EODHD): {news_sentiment_aapl}")

    # Example for testing momentum and integration (using Twitter data if available)
    if twitter_sentiment_btc and twitter_sentiment_btc.get("status") == "success" and twitter_sentiment_btc.get("content_items_analyzed_count",0) > 0:
        dummy_sentiment_series = [
            {"score": twitter_sentiment_btc["avg_polarity"] - 0.1, "timestamp": "T-2"},
            {"score": twitter_sentiment_btc["avg_polarity"] - 0.05, "timestamp": "T-1"},
            {"score": twitter_sentiment_btc["avg_polarity"], "timestamp": "T0"}
        ]
        momentum = sentiment_service.calculate_sentiment_momentum(dummy_sentiment_series)
        print(f"Sentiment Momentum for {btc_symbol} (Twitter): {momentum}")

        dummy_ta_data = {"rsi": 65, "macd_signal": "bullish_cross"}
        combined_analysis = sentiment_service.integrate_sentiment_signals(dummy_ta_data, twitter_sentiment_btc)
        print(f"Combined Analysis for {btc_symbol} (Twitter): {combined_analysis}")
    else:
        print(f"Not enough successful Twitter data for {btc_symbol} for momentum/integration example.")

    education = sentiment_service.get_educational_content()
    print(f"\nEducational Content: {education['title']}")
    print(education['introduction'])
    for concept in education['key_concepts']:
        print(f"- {concept}") 