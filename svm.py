import yfinance as yf

msft = yf.Ticker("MSFT")

# get historical market data
hist = msft.history(period="5y")
direction = hist['Close'] - hist['Open'] > 0

history_len = 64
LR = 0.01
LAMBDA = 0.1

history = [0 for _ in range(history_len)]
weights = [0 for _ in range(history_len + 1)]
correct = 0

# SVM-based branch predictor
for dir in direction:
    # predict and check if correct
    res = -weights[-1]
    for i in range(history_len):
        res += weights[i] * (1 if history[-1 - i] else -1)

    if (res >= 0 and dir) or (res < 0 and not dir):
        correct += 1
    
     
    # update weights and history
    y = 1 if dir else -1
    if y * res >= 1:
        weights[i] -= LR * 2 * LAMBDA * weights[i]
    else:
        for i in range(history_len):
            weights[i] -= LR * (2 * LAMBDA * weights[i] - y * (1 if history[-1 - i] else -1));
        weights[-1] -= LR * y
        
    history.append(dir)
    history = history[1:]

print(correct / len(direction))
