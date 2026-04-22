import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

companies = ["Nvidia", "Apple", "Google", "Microsoft", "Amazon"]
ticks = ["NVDA", "AAPL", "GOOG", "MSFT", "AMZN"]
market_caps = [306.78, 2050, 1360, 1750, 1510]

for company, tick, market_cap in zip(companies, ticks, market_caps):
    ticker = yf.Ticker(tick)
    tickers = yf.Tickers([tick])

    # get historical market data
    history = ticker.history(period="5y", auto_adjust=False)
    history = history["Close"]
    print(history[0])
    history = history / history[0] * market_cap
        
    plt.plot(history.index, history, label=company)

# Add labels, title, and legend for clarity
plt.xlabel("Year")
plt.ylabel("Market Cap (Billions)")
plt.title("Company Market Caps")
plt.legend() # Automatically adds a legend with the specified labels
plt.grid(True)

# Display the plot
plt.show()
