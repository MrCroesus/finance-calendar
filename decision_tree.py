import yfinance as yf
from textblob import TextBlob
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
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
X = X.drop("Open", axis=1).drop("Close", axis=1)

X = X.to_numpy()
test_data = X[-1]
X = X[:-1]
print(X.shape)
y = y.astype(int).to_numpy()
y = y[1:]
print(y.shape)


# all purely quantitative factors in ticker.info except dates, and closely related categorical data
#features = [
#    "auditRisk", "boardRisk", "compensationRisk", "shareHolderRightsRisk", "overallRisk", "governanceEpochDate", "compensationAsOfEpochDate",
#    "86400", "priceHint", "previousClose", "open", "dayLow", "dayHigh",
#    "regularMarketPreviousClose", "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh", "dividendRate", "dividendYield",
#    "exDividendDate", "payoutRatio", "fiveYearAvgDividendYield", "beta", "trailingPE", "forwardPE",
#    "volume", "regularMarketVolume", "averageVolume", "averageVolume10days", "averageDailyVolume10Day",
#    "bid", "ask", "bidSize", "askSize", "marketCap", "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "priceToSalesTrailing12Months", "fiftyDayAverage", "twoHundredDayAverage", "trailingAnnualDividendRate", "trailingAnnualDividendYield", "enterpriseValue", "profitMargins", "floatShares", "sharesOutstanding", "sharesShort", "sharesShortPriorMonth", "sharesShortPreviousMonthDate", "dateShortInterest", "sharesPercentSharesOut", "heldPercentInsiders", "heldPercentInstitutions", "shortRatio", "shortPercentOfFloat", "impliedSharesOutstanding", "bookValue", "priceToBook",  "earningsQuarterlyGrowth", "netIncomeToCommon", "trailingEps", "forwardEps", "enterpriseToRevenue", "enterpriseToEbitda", "52WeekChange", "SandP52WeekChange", "lastDividendValue", "currentPrice", "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice", "recommendationMean", "recommendationKey", "numberOfAnalystOpinions", "totalCash", "totalCashPerShare", "ebitda", "totalDebt", "quickRatio", "currentRatio", "totalRevenue", "debtToEquity", "revenuePerShare", "returnOnAssets", "returnOnEquity", "grossProfits", "freeCashflow", "operatingCashflow", "earningsGrowth", "revenueGrowth", "grossMargins", "ebitdaMargins", "operatingMargins", "fiftyDayAverageChange", "fiftyDayAverageChangePercent", "twoHundredDayAverageChange", "twoHundredDayAverageChangePercent", "sourceInterval", "averageAnalystRating", "regularMarketChangePercent", "regularMarketPrice", "postMarketChangePercent", "postMarketPrice", "postMarketChange", "regularMarketChange", "regularMarketDayRange", "averageDailyVolume3Month", "fiftyTwoWeekLowChange", "fiftyTwoWeekLowChangePercent", "fiftyTwoWeekRange", "fiftyTwoWeekHighChange", "fiftyTwoWeekHighChangePercent", "fiftyTwoWeekChangePercent", "epsTrailingTwelveMonths", "epsForward", "epsCurrentYear", "priceEpsCurrentYear", "trailingPegRatio", "actions", "analyst_price_targets", "balance_sheet", "balancesheet", "calendar", "cashflow", "dividends", "earnings", "earnings_estimate", "earnings_history", "eps_revisions", "eps_trend", "financials", "history"
#]

# date features
# "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter", "lastSplitDate", "lastDividendDate", "postMarketTime", "regularMarketTime", "gmtOffSetMilliseconds", "exchangeDataDelayedBy", "firstTradeDateMilliseconds", "dividendDate", "earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd", "earningsCallTimestampStart", "earningsCallTimestampEnd"

# factors with a Time series
# features for Ticker (or PriceHistory)
features = ["get_actions", "get_capital_gains", "get_dividends", "get_recommendations", "get_shares_full", "get_splits", "history"]

# features for Tickers
features = ["news"]

# features for Search
features = ["news", "research"]



eps = 1e-5  # a small number

def cost(truth, predicted):
    n = len(truth)
    correct = 0
    for true, prediction in zip(truth, predicted):
        if true == prediction:
            correct += 1
    return correct / n
    
def train_split(data, labels, val_fraction):
    n = len(data)

    indices = np.arange(n)
    np.random.shuffle(indices)
    shuffled_training_data = data[indices]
    shuffled_training_labels = labels[indices]

    val_size = int(val_fraction * n)
    return shuffled_training_data[val_size:], shuffled_training_labels[val_size:], shuffled_training_data[:val_size], shuffled_training_labels[:val_size]
    
features_used = []
    
class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO
        counter = Counter(y)

        H = 0
        for count in counter.values():
            p_C = count / len(y)
            H += -p_C * np.log2(p_C + eps)
        return H

    @staticmethod
    def information_gain(X, y, thresh):
        #print(thresh)
        #print(X)
        #print(y)
        
        # TODO
        left_indices = X < thresh
        right_indices = X >= thresh
        y_0 = y[left_indices]
        y_1 = y[right_indices]
        H_after = (len(y_0) * DecisionTree.entropy(y_0) + len(y_1) * DecisionTree.entropy(y_1)) / (len(y_0) + len(y_1))
        return DecisionTree.entropy(y) - H_after

    def split(self, X, y, feature_idx, thresh):
        """
        Split the dataset into two subsets, given a feature and a threshold.
        Return X_0, y_0, X_1, y_1
        where (X_0, y_0) are the subset of examples whose feature_idx-th feature
        is less than thresh, and (X_1, y_1) are the other examples.
        """
        # TODO
        left_indices = X[:, feature_idx] < thresh
        right_indices = X[:, feature_idx] >= thresh
        X_0 = X[left_indices]
        y_0 = y[left_indices]
        X_1 = X[right_indices]
        y_1 = y[right_indices]
        return X_0, y_0, X_1, y_1

    def fit(self, X, y):
        # TODO
        if self.max_depth == 0 or np.all(y == y[0]):
            self.data = X
            self.labels = y
            self.pred = stats.mode(y.astype(int))[0]
            return
        else:
            max_info_gain = 0
            best_idx = -1
            best_thresh = -1
            for idx in range(X.shape[1]):
                X_idx = X[:, idx]

                for thresh in np.unique(X_idx): # use unique feature values as possible thresh values
                    thresh = np.mean(X_idx)
                    info_gain = DecisionTree.information_gain(X_idx, y, thresh)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_idx = idx
                        best_thresh = thresh

            X_0, y_0, X_1, y_1 = self.split(X, y, best_idx, best_thresh)
            self.split_idx = best_idx
            self.thresh = best_thresh

            features_used.append(best_idx)

            if len(y_0) > 0:
                self.left = DecisionTree(self.max_depth - 1, self.features)
                self.left.split_idx = best_idx
                self.left.thresh = best_thresh
                self.left.fit(X_0, y_0)
            if len(y_1) > 0:
                self.right = DecisionTree(self.max_depth - 1, self.features)
                self.right.split_idx = best_idx
                self.right.thresh = best_thresh
                self.right.fit(X_1, y_1)
            return

    def predict(self, X):
        # TODO
        preds = []

        for example in X:
            ptr = self
            while ptr.left or ptr.right:
                if example[ptr.split_idx] >= ptr.thresh:
                    ptr = ptr.right
                else:
                    ptr = ptr.left
            preds.append(ptr.pred)

        return np.array(preds)

    def _to_graphviz(self, node_id):
        if self.max_depth == 0:
            return f'{node_id} [label="Prediction: {self.pred}\nSamples: {self.labels.size}"];\n'
        else:
            graph = f'{node_id} [label="{self.features[self.split_idx]} < {self.thresh:.2f}"];\n'
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            if self.left is not None:
                graph += f'{node_id} -> {left_id};\n'
                graph += self.left._to_graphviz(left_id)
            if self.right is not None:
                graph += f'{node_id} -> {right_id};\n'
                graph += self.right._to_graphviz(right_id)
            return graph

    def to_graphviz(self):
        graph = "digraph Tree {\nnode [shape=box];\n"
        graph += self._to_graphviz(0)
        graph += "}\n"
        return graph
        
    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())
                                           
                                           
class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO
        for i in range(self.n):
            sample_idx = np.random.choice(np.arange(X.shape[0]), int(1.172 * self.n), replace=True)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]

            self.decision_trees[i].fit(X_sample, y_sample)
        return

    def predict(self, X):
        # TODO
        predictions = [decision_tree.predict(X) for decision_tree in self.decision_trees]
        return stats.mode(predictions)[0][0]


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)
        
        

X, y, validation_data, validation_labels = train_split(X, y, 0.20)
validation_accs = []

# decision tree stock predictor
for i in range(1, 41):
    print("\n\nDecision Tree")
    dt = DecisionTree(max_depth=i, feature_labels=features)
    dt.fit(X, y)
    pred = dt.predict(validation_data)
    print(cost(validation_labels, pred))
    validation_accs.append(cost(validation_labels, pred))

print(np.max(validation_accs))
plt.plot(np.arange(1, 41), validation_accs)
plt.show()
