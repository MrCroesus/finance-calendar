import yfinance as yf
import numpy as np

msft = yf.Ticker("MSFT")

# get historical market data
hist = msft.history(period="5y")
direction = hist['Close'] - hist['Open'] > 0

bits = 3
threshold = 1 << (bits - 1)
counter = threshold
correct = 0

# branch history table with single 2-bit counter
for dir in direction:
    if (counter >= threshold and dir) or (counter < threshold and not dir):
        correct += 1
        
    if dir:
        counter = min(1 << bits, counter + 1)
    else:
        counter = max(0, counter - 1)

print(correct / len(direction))
