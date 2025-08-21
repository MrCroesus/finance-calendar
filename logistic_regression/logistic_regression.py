import yfinance as yf
import math

msft = yf.Ticker("MSFT")

# get historical market data
hist = msft.history(period="5y")
direction = hist['Close'] - hist['Open'] > 0

history_len = 64
LR = 0.1

def constrain(value):
    return min(max_weight, max(min_weight, value))

history = [0 for _ in range(history_len)]
weights = [0 for _ in range(history_len + 1)]
correct = 0

# logistic regression branch predictor
for dir in direction:
    # predict and check if correct
    res = weights[-1]
    for i in range(history_len):
        res += weights[i] * (1 if history[-1 - i] else -1)
    
    pred = 1 / (1 + math.exp(-res)) >= 0.5
    if (pred and dir) or (not pred and not dir):
        correct += 1
     
     
    # update weights and history
    y = 1 if dir else -1
    temp = pred - y
    for i in range(history_len):
        weights[i] -= LR * temp * (1 if history[-1 - i] else -1)
    weights[-1] -= LR * temp
        
    history.append(dir)
    history = history[1:]

print(correct / len(direction))

