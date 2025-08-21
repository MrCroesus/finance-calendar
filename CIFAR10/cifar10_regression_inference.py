import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import matplotlib.pyplot as plt

companies = ["Nvidia", "Microsoft", "Apple"]
ticks = ["NVDA", "MSFT", "AAPL"]
preds = {}
for company, tick in zip(companies, ticks):
    ticker = yf.Ticker(tick)
    tickers = yf.Tickers([tick])

    # get historical market data
    history = ticker.history(period="1y")
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
    print(news_sentiments)
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
    X = np.where(X == 0, 0.01, X)
    X = np.divide(X, np.mean(X, axis=0, keepdims=True), out=np.zeros_like(X, dtype=float))
    testing_data = X[-1]
    X = X[:-1]
    print(X.shape)
    y = y.astype(int).to_numpy()
    y = y[1:]
    print(y.shape)



    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Using device", device)


    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(7, 1024)
            self.dropout1 = nn.Dropout(0.2)
            self.linear2 = nn.Linear(1024, 1024)
            self.dropout2 = nn.Dropout(0.2)
            self.linear3 = nn.Linear(1024, 1)

        def forward(self, x):
            x = self.linear1(x)
            x = nn.ReLU()(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = nn.ReLU()(x)
            x = self.dropout2(x)
            x = self.linear3(x)
            return x

    model = NeuralNetwork().to(device)

    best_model_params_path = '/Users/harristhai/finance-calendar/CIFAR10/best_cifar10_regression_params.pt'
    model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    model.eval()
    pred = model(torch.from_numpy(testing_data).float().to(device))
    print(pred)
    preds[tick] = round(pred.item(), 2)

print(preds)
