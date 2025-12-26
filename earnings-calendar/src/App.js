import React, { useState, useEffect, useRef } from 'react';
import { Search, X, TrendingUp, RefreshCw, Check } from 'lucide-react';

export default function StockTickerManager() {
  const [tickers, setTickers] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [allStocks, setAllStocks] = useState([]);
  const [isLoadingStocks, setIsLoadingStocks] = useState(true);
  const dropdownRef = useRef(null);
  
  // Replace with your Finnhub API key (get free at https://finnhub.io)
  const FINNHUB_API_KEY = 'd56v729r01qkvkasp1tgd56v729r01qkvkasp1u0';

  // Common company name aliases (old names, nicknames, etc.)
  const COMPANY_ALIASES = {
    'META': ['facebook', 'fb', 'instagram', 'whatsapp'],
    'GOOGL': ['google', 'alphabet'],
    'GOOG': ['google', 'alphabet'],
    'BRK.B': ['berkshire', 'berkshire hathaway', 'buffett'],
    'BRK.A': ['berkshire', 'berkshire hathaway', 'buffett'],
    'TSLA': ['tesla', 'elon'],
    'AAPL': ['apple', 'iphone', 'mac'],
    'MSFT': ['microsoft', 'windows'],
    'AMZN': ['amazon', 'aws'],
    'NFLX': ['netflix'],
    'DIS': ['disney', 'walt disney'],
    'V': ['visa'],
    'MA': ['mastercard'],
    'NVDA': ['nvidia'],
    'AMD': ['amd', 'advanced micro'],
    'INTC': ['intel'],
    'NSRGY': ['nestle', 'nestlé'],
    'TOYOF': ['toyota'],
    'SMAWF': ['samsung'],
    'RHHBY': ['roche'],
    'AUDVF': ['adidas'],
  };

  // Load all stock symbols on component mount
  useEffect(() => {
    const loadAllStocks = async () => {
      try {
        setIsLoadingStocks(true);
        
        // Fetch from multiple exchanges to get broader coverage
        const exchanges = ['US', 'OTC'];  // US exchanges + OTC (international ADRs)
        const allStockData = [];
        
        for (const exchange of exchanges) {
          try {
            const response = await fetch(
              `https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${FINNHUB_API_KEY}`
            );
            
            if (response.ok) {
              const data = await response.json();
              allStockData.push(...data);
            }
          } catch (error) {
            console.error(`Error loading ${exchange} stocks:`, error);
          }
        }
        
        // Filter and deduplicate stocks
        const uniqueSymbols = new Set();
        const stocks = allStockData
          .filter(stock => {
            if (!stock.symbol || !stock.description) {
              return false;
            }
            
            // Skip duplicates
            if (uniqueSymbols.has(stock.symbol)) {
              return false;
            }
            
            // Keep common stocks and ADRs
            const validTypes = ['Common Stock', 'ADR', 'GDR'];
            if (!validTypes.includes(stock.type)) {
              return false;
            }
            
            uniqueSymbols.add(stock.symbol);
            return true;
          })
          .map(stock => ({
            symbol: stock.symbol,
            description: stock.description,
            type: stock.type
          }));
        
        setAllStocks(stocks);
        console.log(`Loaded ${stocks.length} stocks from ${exchanges.join(', ')} for searching`);
      } catch (error) {
        console.error('Error loading stocks:', error);
      } finally {
        setIsLoadingStocks(false);
      }
    };

    if (FINNHUB_API_KEY !== 'your_api_key_here') {
      loadAllStocks();
    } else {
      setIsLoadingStocks(false);
    }
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Calculate Levenshtein distance for typo tolerance
  const levenshteinDistance = (str1, str2) => {
    const len1 = str1.length;
    const len2 = str2.length;
    const matrix = [];

    for (let i = 0; i <= len1; i++) {
      matrix[i] = [i];
    }

    for (let j = 0; j <= len2; j++) {
      matrix[0][j] = j;
    }

    for (let i = 1; i <= len1; i++) {
      for (let j = 1; j <= len2; j++) {
        if (str1[i - 1] === str2[j - 1]) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }

    return matrix[len1][len2];
  };

  // Enhanced fuzzy scoring with multiple strategies
  const fuzzyScore = (text, query, symbol) => {
    if (!text || !query) return 0;
    
    text = text.toLowerCase();
    query = query.toLowerCase();
    
    let score = 0;
    
    // Check aliases first (exact match on alias)
    if (symbol && COMPANY_ALIASES[symbol]) {
      const aliases = COMPANY_ALIASES[symbol];
      for (const alias of aliases) {
        if (alias === query) {
          return 15000; // Higher than exact match
        }
        if (alias.includes(query) || query.includes(alias)) {
          score += 8000;
        }
      }
    }
    
    // Strategy 1: Exact match (highest priority)
    if (text === query) return 10000;
    
    // Strategy 2: Starts with (very high priority)
    if (text.startsWith(query)) return 5000;
    
    // Strategy 3: Word starts with (high priority)
    const words = text.split(/\s+/);
    for (const word of words) {
      if (word.startsWith(query)) {
        score += 3000;
      }
    }
    
    // Strategy 4: Contains query (medium priority)
    if (text.includes(query)) {
      score += 1000;
    }
    
    // Strategy 5: Levenshtein distance for typos
    for (const word of words) {
      // Check if query is similar to this word
      const compareLength = Math.min(word.length, query.length);
      const distance = levenshteinDistance(
        word.substring(0, compareLength + 2), 
        query
      );
      
      // Allow 1-2 character differences
      const tolerance = query.length <= 4 ? 1 : 2;
      if (distance <= tolerance) {
        score += 500 - (distance * 100);
      }
    }
    
    // Strategy 6: Character sequence matching (for abbreviations)
    let queryIndex = 0;
    let consecutiveMatches = 0;
    
    for (let i = 0; i < text.length && queryIndex < query.length; i++) {
      if (text[i] === query[queryIndex]) {
        score += 10;
        consecutiveMatches++;
        queryIndex++;
        
        // Bonus for consecutive matches
        if (consecutiveMatches > 1) {
          score += 5 * consecutiveMatches;
        }
      } else {
        consecutiveMatches = 0;
      }
    }
    
    // Bonus if all characters matched in order
    if (queryIndex === query.length) {
      score += 100;
    }
    
    return score;
  };

  // Search locally through all stocks
  useEffect(() => {
    const searchLocally = () => {
      if (inputValue.trim().length < 1) {
        setSearchResults([]);
        setShowDropdown(false);
        setSelectedIndex(0);
        return;
      }

      if (allStocks.length === 0) {
        return;
      }

      setIsSearching(true);
      setShowDropdown(true);

      // Score all stocks (including already-added tickers)
      const scoredResults = allStocks
        .map(stock => {
          const symbolScore = fuzzyScore(stock.symbol, inputValue, stock.symbol);
          const descScore = fuzzyScore(stock.description, inputValue, stock.symbol);
          const maxScore = Math.max(symbolScore, descScore);
          
          // Check if this ticker is already added (just for display, doesn't affect score)
          const isAdded = tickers.includes(stock.symbol);
          
          return {
            symbol: stock.symbol,
            description: stock.description,
            score: maxScore,
            isAdded: isAdded
          };
        })
        .filter(stock => stock.score > 0)
        .sort((a, b) => b.score - a.score) // Sort by score only
        .slice(0, 10); // Take top 10

      console.log('Tickers array:', tickers);
      console.log('Total results before top 10:', scoredResults.length);
      console.log('Search results (top 10):', scoredResults.map(r => ({ 
        symbol: r.symbol, 
        isAdded: r.isAdded,
        score: r.score
      })));
      
      setSearchResults(scoredResults);
      setSelectedIndex(0);
      setIsSearching(false);
    };

    const debounceTimer = setTimeout(searchLocally, 150);
    return () => clearTimeout(debounceTimer);
  }, [inputValue, allStocks, tickers, FINNHUB_API_KEY]); // Added tickers and FINNHUB_API_KEY to dependencies

  const addTicker = (symbol, description = '') => {
    const ticker = symbol.toUpperCase();
    
    if (!ticker || tickers.includes(ticker)) {
      return;
    }
    
    setTickers([...tickers, ticker]);
    setInputValue('');
    setSearchResults([]);
    setShowDropdown(false);
  };

  const removeTicker = (tickerToRemove) => {
    setTickers(tickers.filter(t => t !== tickerToRemove));
  };

  const handleKeyDown = (e) => {
    // Only handle these keys when dropdown is open
    if (!showDropdown || searchResults.length === 0) {
      if (e.key === 'Enter' && inputValue.trim()) {
        // Add manually typed ticker if no results
        addTicker(inputValue);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => 
        prev < searchResults.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => prev > 0 ? prev - 1 : 0);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      // Add the selected result
      const selected = searchResults[selectedIndex];
      if (selected && !selected.isAdded) {
        addTicker(selected.symbol, selected.description);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setShowDropdown(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && searchResults.length > 0) {
      addTicker(searchResults[0].symbol, searchResults[0].description);
    } else if (e.key === 'Enter') {
      addTicker(inputValue);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="flex items-center gap-3 mb-6">
            <TrendingUp className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">
              Stock Ticker Manager
            </h1>
          </div>

          {/* Loading indicator */}
          {isLoadingStocks && (
            <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-400 rounded flex items-center gap-3">
              <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
              <p className="text-sm text-blue-800">
                Loading stock database for fuzzy search...
              </p>
            </div>
          )}

          {/* Stock count indicator */}
          {!isLoadingStocks && allStocks.length > 0 && (
            <div className="mb-4 text-xs text-gray-500">
              Searching across {allStocks.length.toLocaleString()} stocks (US & International)
            </div>
          )}
          
          {/* Search Bar with Dropdown */}
          <div className="relative mb-6" ref={dropdownRef}>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                onFocus={() => inputValue && setShowDropdown(true)}
                placeholder="Search by ticker or company name (US & international stocks)..."
                disabled={isLoadingStocks || allStocks.length === 0}
                className="w-full pl-12 pr-4 py-4 text-lg border-2 border-gray-200 rounded-xl focus:border-indigo-500 focus:outline-none transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
            </div>

            {/* Search Results Dropdown */}
            {showDropdown && searchResults.length > 0 && !isSearching && (
              <div className="absolute z-10 w-full mt-2 bg-white rounded-xl shadow-lg border border-gray-200 max-h-96 overflow-y-auto">
                {searchResults.map((result, index) => (
                  <button
                    key={result.symbol}
                    onClick={() => !result.isAdded && addTicker(result.symbol, result.description)}
                    disabled={result.isAdded}
                    className={`w-full px-4 py-3 text-left transition-colors border-b border-gray-100 last:border-b-0 flex items-start gap-3 ${
                      index === selectedIndex 
                        ? 'bg-indigo-100' 
                        : result.isAdded 
                        ? 'bg-gray-50 cursor-default' 
                        : 'hover:bg-indigo-50 cursor-pointer'
                    }`}
                  >
                    <div className={`flex-shrink-0 px-2 py-1 rounded font-bold text-sm mt-0.5 ${
                      result.isAdded 
                        ? 'bg-green-100 text-green-700' 
                        : 'bg-indigo-100 text-indigo-700'
                    }`}>
                      {result.symbol}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`font-medium truncate ${
                        result.isAdded ? 'text-gray-500' : 'text-gray-800'
                      }`}>
                        {result.description}
                      </p>
                    </div>
                    {result.isAdded && (
                      <Check className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                    )}
                  </button>
                ))}
              </div>
            )}

            {/* Searching state */}
            {isSearching && showDropdown && (
              <div className="absolute z-10 w-full mt-2 bg-white rounded-xl shadow-lg border border-gray-200 p-4 text-center text-gray-500">
                Searching...
              </div>
            )}

            {/* No results */}
            {showDropdown && !isSearching && inputValue && searchResults.length === 0 && allStocks.length > 0 && (
              <div className="absolute z-10 w-full mt-2 bg-white rounded-xl shadow-lg border border-gray-200 p-4 text-center text-gray-500">
                No stocks found. Try a different search.
              </div>
            )}
          </div>

          {/* API Key Notice */}
          {FINNHUB_API_KEY === 'your_api_key_here' && (
            <div className="mb-6 p-4 bg-yellow-50 border-l-4 border-yellow-400 rounded">
              <p className="text-sm text-yellow-800">
                <strong>Note:</strong> You need a Finnhub API key for search to work. 
                Get a free key at <a href="https://finnhub.io" target="_blank" rel="noopener noreferrer" className="underline">finnhub.io</a> and 
                replace <code className="bg-yellow-100 px-1 rounded">FINNHUB_API_KEY</code> in the code.
              </p>
            </div>
          )}
          
          {/* Ticker Pills */}
          {tickers.length > 0 && (
            <div className="flex flex-wrap gap-3 mb-6">
              {tickers.map(ticker => (
                <div
                  key={ticker}
                  className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2 rounded-full text-sm font-medium shadow-md hover:bg-indigo-700 transition-colors"
                >
                  <span>{ticker}</span>
                  <button
                    onClick={() => removeTicker(ticker)}
                    className="hover:bg-indigo-800 rounded-full p-1 transition-colors"
                    aria-label={`Remove ${ticker}`}
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
          
          {/* Empty State */}
          {tickers.length === 0 && !isLoadingStocks && (
            <div className="text-center py-12">
              <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-500 text-lg">
                Search for stocks to get started
              </p>
              <p className="text-gray-400 text-sm mt-2">
                Type a ticker symbol or company name (typos are OK!)
              </p>
            </div>
          )}
          
          {/* Ticker Count */}
          {tickers.length > 0 && (
            <div className="pt-6 border-t border-gray-200">
              <p className="text-gray-600 text-sm">
                <span className="font-semibold">{tickers.length}</span> ticker{tickers.length !== 1 ? 's' : ''} added
              </p>
            </div>
          )}
        </div>

        {/* Fuzzy Search Examples */}
        {!isLoadingStocks && allStocks.length > 0 && tickers.length === 0 && (
          <div className="mt-6 bg-white rounded-xl shadow-md p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Try these searches:</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Type:</span> <code className="text-indigo-600 font-mono">nestle</code> → finds NSRGY
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Type:</span> <code className="text-indigo-600 font-mono">facebook</code> → finds META
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Type:</span> <code className="text-indigo-600 font-mono">toyota</code> → finds TOYOF
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Type:</span> <code className="text-indigo-600 font-mono">berkshire</code> → finds BRK.B
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}