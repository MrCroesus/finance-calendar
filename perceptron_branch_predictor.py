import yfinance as yf

msft = yf.Ticker("MSFT")

# get historical market data
hist = msft.history(period="5y")
direction = hist['Close'] - hist['Open'] > 0

history_len = 48
threshold = 1.93 * history_len + 14
max_weight = 127
min_weight = -128
def constrain(value):
    return min(max_weight, max(min_weight, value))

history = [0 for _ in range(history_len)]
perceptron = [0 for _ in range(history_len + 1)]
correct = 0

# perceptron branch predictor
for dir in direction:
    # predict and check if correct
    res = perceptron[0]
    for i in range(history_len):
        res += perceptron[i + 1] * (1 if history[-1 - i] else -1)

    if (res >= 0 and dir) or (res < 0 and not dir):
        correct += 1
     
     
    # update perceptron and history
    y = 1 if dir else -1
    if y * res <= 0 or y * res < threshold:
        perceptron[0] = constrain(perceptron[0] + y)
        
        for i in range(history_len):
          perceptron[i + 1] = constrain(perceptron[i + 1] + y * (1 if history[-1 - i] else -1))
        
    history.append(dir)
    history = history[1:]

print(correct / len(direction))
