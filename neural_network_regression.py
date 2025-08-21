import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
y = (X["Close"] - X["Open"]) / X["Open"] * 100
X = X.drop("Dividends", axis=1).drop("Stock Splits", axis=1)
print(X.keys())

X = X.to_numpy()
test_data = X[-1]
X = X[:-1]
print(X.shape)
y = y.astype(float).to_numpy()
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

X = np.divide(X, np.mean(X, axis=0, keepdims=True), out=np.zeros_like(X, dtype=float), where=X != 0)
train_data, train_labels, test_data, test_labels = train_split(X, y, 0.20)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)


train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
for x, y in train_dataloader:
    print("Batch x:", x)
    print("Batch y:", y)
    break
    
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)
for x, y in test_dataloader:
    print("Batch x:", x)
    print("Batch y:", y)
    break


# ### YOUR CODE HERE ###
batch_size = 64

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3)
        self.conv2 = nn.Conv1d(64, 64, 3)
        self.linear1 = nn.Linear(32, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        x = x.float()
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        return x

model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            train_loss = loss
            print(f"loss: {loss:>7f}")
    return train_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

epochs = 8
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    test_loss = test(test_dataloader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
print("Done!")

#x = np.arange(epochs)
#plt.plot(x, train_losses, label = "training loss")
#plt.plot(x, test_losses, label = "validation loss")
#plt.legend()
#plt.show()
