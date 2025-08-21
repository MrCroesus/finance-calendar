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
X = X[:-1]
print(X.shape)
y = y.astype(int).to_numpy()
y = y[1:]
print(y.shape)

X = np.divide(X, np.mean(X, axis=0, keepdims=True), out=np.zeros_like(X, dtype=float), where=X != 0)

def train_split(data, labels, val_fraction):
    n = len(data)

    indices = np.arange(n)
    np.random.shuffle(indices)
    shuffled_training_data = data[indices]
    shuffled_training_labels = labels[indices]

    val_size = int(val_fraction * n)
    return shuffled_training_data[val_size:], shuffled_training_labels[val_size:], shuffled_training_data[:val_size], shuffled_training_labels[:val_size]

train_data, train_labels, val_data, val_labels = train_split(X, y, 0.20)



device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Using device", device)


train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
for x, y in train_dataloader:
    print("Batch x:", x)
    print("Batch y:", y)
    break
    
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
for x, y in val_dataloader:
    print("Batch x:", x)
    print("Batch y:", y)
    break
    
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


# ### YOUR CODE HERE ###
batch_size = 64

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 1024)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = x.float()
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x

model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


training_losses = []
validation_losses = []
training_accuracies = []
validation_accuracies = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_params_path = '/Users/harristhai/finance-calendar/CIFAR10/cifar10_params.pt'

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train' and epoch > 0:
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' and epoch > 0):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(preds)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch > 0:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and epoch > 0:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            if phase == 'train':
                training_losses.append(epoch_loss)
                training_accuracies.append(float(epoch_acc))
            else:
                validation_losses.append(epoch_loss)
                validation_accuracies.append(float(epoch_acc))
#            print(training_losses, training_accuracies, validation_losses, validation_accuracies)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model
    
    
num_epochs = 100

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)


#def test(dataloader, model, loss_fn):
#    size = len(dataloader.dataset)
#    num_batches = len(dataloader)
#    model.eval()
#    test_loss, correct = 0, 0
#    with torch.no_grad():
#        for X, y in dataloader:
#            X, y = X.to(device), y.to(device)
#            pred = model(X)
#            test_loss += loss_fn(pred, y).item()
#            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#    test_loss /= num_batches
#    correct /= size
#    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
#    return test_loss, correct
#
#epochs = 8
#train_losses = []
#train_accuracies = []
#test_losses = []
#test_accuracies = []
#for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------------")
#    train_loss, train_correct = train(train_dataloader, model, loss_fn, optimizer)
#    test_loss, test_correct = test(test_dataloader, model, loss_fn)
#    train_losses.append(train_loss)
#    train_accuracies.append(train_correct)
#    test_losses.append(test_loss)
#    test_accuracies.append(test_correct)
#print("Done!")

x = np.arange(num_epochs + 1)
plt.clf()
plt.plot(x, training_losses, label = "training loss")
plt.plot(x, validation_losses, label = "validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
plt.plot(x, training_accuracies, label = "training accuracy")
plt.plot(x, validation_accuracies, label = "validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
