import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

company = "Microsoft"
tick = "MSFT"
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
X = X / np.mean(X, axis=0, keepdims=True)
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


knn = NearestNeighbors(n_neighbors=3).fit(train_data)
indices = knn.kneighbors(np.reshape(testing_data, (1, len(testing_data))), n_neighbors=3,return_distance=False)
print(testing_data)
print(train_data[indices])
print(train_labels[indices])
print(np.sum(train_labels[indices]) > len(indices) / 2)


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

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error')
        plt.title('Mean Displacement Error vs. k in kNN')
        plt.show()

    return ks[np.where(mean_errors == min(mean_errors))[0][0]]
    
    
grid_search(train_data, train_labels, test_data, test_labels)


k = grid_search(train_data, train_labels, test_data, test_labels, is_weighted=True)
print("k =", k)


mean_errors_lin = []
mean_errors_nn = []
ratios = np.arange(0.1, 1.1, 0.1)
for r in ratios:
    num_samples = int(r * len(train_data))
    ##### TODO(g): Your Code Here #####
    indices = np.random.choice(np.arange(len(train_data)), num_samples)
    new_train_data = train_data[indices]
    new_train_labels = train_labels[indices]
    
    lin = LinearRegression().fit(new_train_data, new_train_labels)
    lin_preds = lin.predict(test_data)
    lin_preds = lin_preds - test_labels
    lin_preds = np.linalg.norm(lin_preds)
    
    knn = NearestNeighbors(n_neighbors=100).fit(new_train_data)
    indices = knn.kneighbors(test_data, n_neighbors=k,return_distance=False)

    nn_preds = train_labels[indices[:, 0]] - test_labels
    nn_preds = np.linalg.norm(nn_preds)
    
    e_lin = np.mean(lin_preds)
    e_nn = np.mean(nn_preds)

    mean_errors_lin.append(e_lin)
    mean_errors_nn.append(e_nn)

    print(f'\nTraining set ratio: {r:.1f} ({num_samples})')
    print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
    print(f'kNN mean displacement error (miles): {e_nn:.1f}')

# Plot error vs training set size
plt.plot(ratios, mean_errors_lin, label='lin. reg.')
plt.plot(ratios, mean_errors_nn, label='kNN')
plt.xlabel('Training Set Ratio')
plt.ylabel('Mean Displacement Error (miles)')
plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
plt.legend()
plt.show()
