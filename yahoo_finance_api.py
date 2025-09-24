import yfinance as yf

msft = yf.Ticker("MSFT")
tickers = yf.Tickers(["MSFT"])
#download = tickers.download(period='1mo')
#print(download.keys())

# get all stock info
#msft.info
print(msft.eps_revisions)

# get historical market data
#hist = msft.history(period="1mo")
#actions = msft.get_actions(period="max")
#print(actions)

# show meta information about the history (requires history() to be called first)
#msft.history_metadata

# show actions (dividends, splits, capital gains)
#msft.actions
#msft.funds_data
#msft.dividends
#msft.splits
#msft.capital_gains  # only for mutual funds & etfs
#
## show share count
#msft.get_shares_full(start="2022-01-01", end=None)
#
## show financials:
## - income statement
#msft.income_stmt
#msft.quarterly_income_stmt
## - balance sheet
#msft.balance_sheet
#msft.quarterly_balance_sheet
## - cash flow statement
#msft.cashflow
#msft.quarterly_cashflow
## see `Ticker.get_income_stmt()` for more options
#
## show holders
#msft.major_holders
#msft.institutional_holders
#msft.mutualfund_holders
#msft.insider_transactions
#msft.insider_purchases
#msft.insider_roster_holders
#
## show recommendations
#msft.recommendations
#msft.recommendations_summary
#msft.upgrades_downgrades
#
## Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default.
## Note: If more are needed use msft.get_earnings_dates(limit=XX) with increased limit argument.
#msft.earnings_dates
#
## show ISIN code - *experimental*
## ISIN = International Securities Identification Number
#msft.isin
#
## show options expirations
#msft.options
#
## show news
#msft.news
#
## get option chain for specific expiration
#opt = msft.option_chain('YYYY-MM-DD')
## data available via: opt.calls, opt.puts


# get list of quotes
quotes = yf.Search("AAPL", max_results=10).quotes

# get list of news
news = yf.Search("Google", news_count=10).news

# get list of related research
research = yf.Search("apple", include_research=True).research

# Get All
all = yf.Lookup("AAPL").all
all = yf.Lookup("AAPL").get_all(count=100)

# Get Stocks
stock = yf.Lookup("AAPL").stock
stock = yf.Lookup("AAPL").get_stock(count=100)

# Get Mutual Funds
mutualfund = yf.Lookup("AAPL").mutualfund
mutualfund = yf.Lookup("AAPL").get_mutualfund(count=100)

# Get ETFs
etf = yf.Lookup("AAPL").etf
etf = yf.Lookup("AAPL").get_etf(count=100)

# Get Indices
index = yf.Lookup("AAPL").index
index = yf.Lookup("AAPL").get_index(count=100)

# Get Futures
future = yf.Lookup("AAPL").future
future = yf.Lookup("AAPL").get_future(count=100)

# Get Currencies
currency = yf.Lookup("AAPL").currency
currency = yf.Lookup("AAPL").get_currency(count=100)

# Get Cryptocurrencies
cryptocurrency = yf.Lookup("AAPL").cryptocurrency
cryptocurrency = yf.Lookup("AAPL").get_cryptocurrency(count=100)
