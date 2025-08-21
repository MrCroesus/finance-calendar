import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

companies = ["Nvidia", "Microsoft", "Apple"]
ticks = ["NVDA", "MSFT", "AAPL"]
preds = {}
for company, tick in zip(companies, ticks):
    ticker = yf.Ticker(tick)
    tickers = yf.Tickers([tick])

    # get historical market data
    history = ticker.history(period="5y")
    history.index = history.index.date
    print(history.keys())

    def news_sentiment(content):
        return TextBlob(content['title']).sentiment.polarity + TextBlob(content['description']).sentiment.polarity + TextBlob(content['summary']).sentiment.polarity
        
    def research_sentiment(content):
        return TextBlob(content).sentiment.polarity

    # get stock news
    news = tickers.news()
    news_sentiments_map = {}
    for article in news[tick]:
        content = article['content']
        date = content['pubDate'][:10]
        if date not in news_sentiments_map:
            news_sentiments_map[date] = 0
        news_sentiments_map[date] += news_sentiment(content)
    news_sentiments_dict = {"Date": news_sentiments_map.keys(), "News Sentiment": news_sentiments_map.values()}
    news_sentiments = pd.DataFrame.from_dict(news_sentiments_dict)
    news_sentiments.set_index('Date', inplace=True)
    news_sentiments.index = pd.to_datetime(news_sentiments.index)
    print(news_sentiments.keys())

    # get company research
    research = yf.Search(company, include_research=True).research
    research_sentiments_map = {}
    for article in research:
        content = article['reportHeadline']
        date = article['reportDate'] // 86400 * 86400
        if date not in news_sentiments_map:
            research_sentiments_map[date] = 0
        research_sentiments_map[date] += research_sentiment(content)
    research_sentiments_dict = {"Date": research_sentiments_map.keys(), "Research Sentiment": research_sentiments_map.values()}
    research_sentiments = pd.DataFrame.from_dict(research_sentiments_dict)
    research_sentiments.set_index('Date', inplace=True)
    research_sentiments.index = pd.to_datetime(research_sentiments.index, unit='ms').date
    print(research_sentiments.keys())

    X = history.join(news_sentiments, how='outer').join(research_sentiments, how='outer')
    X = X.interpolate().fillna(method='bfill')
    y = X["Close"] > X["Open"]
    X = X.drop("Dividends", axis=1).drop("Stock Splits", axis=1)
    print(X.keys())

    X = X.to_numpy()
    means = np.mean(X, axis=0, keepdims=True)
    means[means == 0] = 1
    X = X / means
    testing_data = X[-1]
    X = X[:-1]
    print(X.shape)
    y = y.astype(int).to_numpy()
    y = y[1:]
    print(y.shape)

    def train_split(data, labels, val_fraction):
        n = len(data)

        indices = np.arange(n)
        np.random.shuffle(indices)
        shuffled_training_data = data[indices]
        shuffled_training_labels = labels[indices]

        val_size = int(val_fraction * n)
        return shuffled_training_data[val_size:], shuffled_training_labels[val_size:], shuffled_training_data[:val_size], shuffled_training_labels[:val_size]

    train_data, train_labels, test_data, test_labels = train_split(X, y, 0.20)


    def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
        """
        Input:
            train_features: Training set image features
            train_labels: Training set GPS (lat, lon) coords
            test_features: Test set image features
            test_labels: Test set GPS (lat, lon) coords
            is_weighted: Weight prediction by distances in feature space

        Output:
            Prints mean displacement error as a function of k
            Plots mean displacement error vs k

        Returns:
            Minimum mean displacement error
        """
        # Evaluate mean displacement error of kNN regression for different values of k
        knn = NearestNeighbors(n_neighbors=100).fit(train_features)

        if verbose:
            print(f'Running grid search for k (is_weighted={is_weighted})')

        ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
        mean_errors = []
        for k in ks:
            distances, indices = knn.kneighbors(test_features, n_neighbors=k)

            errors = []
            for i, nearest in enumerate(indices):
                y = test_labels[i]

                ##### TODO(d): Your Code Here #####
                error_coords = train_labels[nearest] - y
                
                ##### TODO(f): Modify Your Code #####
                e = np.linalg.norm(error_coords)
                if is_weighted:
                    e = e / (distances[i] + 1e-8)

                errors.append(e)
            
            mean_error = np.mean(np.array(errors))
            mean_errors.append(mean_error)
            if verbose:
                print(f'{k}-NN mean displacement error: {mean_error:.1f}')

        return ks[np.where(mean_errors == min(mean_errors))[0][0]]
        

    k = grid_search(train_data, train_labels, test_data, test_labels, is_weighted=True)
    print("k =", k)

    r = 0.65
    num_samples = int(r * len(train_data))
    ##### TODO(g): Your Code Here #####
    indices = np.random.choice(np.arange(len(train_data)), num_samples)
    new_train_data = train_data[indices]
    new_train_labels = train_labels[indices]
    
    lin = LinearRegression().fit(new_train_data, new_train_labels)
    pred = lin.predict(testing_data.reshape(1, len(testing_data)))
    print(pred)
    preds[tick] = "Up" if np.sum(pred) / 2 >= len(pred) else "Down"

print(preds)
