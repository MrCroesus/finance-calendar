import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm.notebook import tqdm

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
X = X.drop("Open", axis=1).drop("Close", axis=1)

X = X.to_numpy()
test_data = X[-1]
X = X[:-1]
print(X.shape)
y = y.astype(int).to_numpy()
y = y[1:]
print(y.shape)

X = np.divide(X, np.mean(X, axis=0, keepdims=True), out=np.zeros_like(X, dtype=float), where=X != 0)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", device)


class Perceptron(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    self.layer = nn.Linear(in_dim, 1) # This is a linear layer, it computes Xw + b

  def forward(self, x):
    return torch.sigmoid(self.layer(x)).squeeze(-1)

perceptron = Perceptron(10)
perceptron = perceptron.to(device) # Move all the perceptron's tensors to the device
print("Parameters", list(perceptron.parameters()))


dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
for x, y in dataloader:
    print("Batch x:", x)
    print("Batch y:", y)
    break


epochs = 10
batch_size = 10
learning_rate = 0.01

num_features = dataset[0][0].shape[0]
model = Perceptron(num_features).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCELoss()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train() # Put model in training mode
for epoch in range(epochs):
    training_losses = []
    for x, y in tqdm(dataloader, unit="batch"):
        x, y = x.float().to(device), y.float().to(device)
        optimizer.zero_grad() # Remove the gradients from the previous step
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
    print("Finished Epoch", epoch + 1, ", training loss:", np.mean(training_losses))

# We can run predictions on the data to determine the final accuracy.
with torch.no_grad():
    model.eval() # Put model in eval mode
    num_correct = 0
    for x, y in dataloader:
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        num_correct += torch.sum(torch.round(pred) == y).item()
    print("Final Accuracy:", num_correct / len(dataset))
    model.train() # Put model back in train mode
