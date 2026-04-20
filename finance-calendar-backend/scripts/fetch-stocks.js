// scripts/fetch-stocks.js
import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';

// Load env vars
if (!process.env.SUPABASE_URL) {
  try {
    const envContent = await fs.readFile('.env', 'utf-8');
    envContent.split('\n').forEach(line => {
      const [key, ...valueParts] = line.split('=');
      if (key && valueParts.length) {
        process.env[key.trim()] = valueParts.join('=').trim();
      }
    });
  } catch (err) {
    console.error('⚠️  No .env file found');
  }
}

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SECRET_KEY
);

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;

async function fetchAndCacheStocks() {
  console.log('🔄 Fetching stocks from Finnhub...\n');

  try {
    const exchanges = ['US', 'OTC'];

    const fetchPromises = exchanges.map(exchange =>
      fetch(`https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${FINNHUB_API_KEY}`)
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .catch(err => {
          console.error(`Error loading ${exchange} stocks:`, err);
          return [];
        })
    );

    const results = await Promise.all(fetchPromises);
    const allStockData = results.flat();

    const uniqueSymbols = new Set();
    const stocks = allStockData
      .filter(stock => {
        if (!stock.symbol || !stock.description) return false;
        if (uniqueSymbols.has(stock.symbol)) return false;
        const validTypes = ['Common Stock', 'ADR', 'GDR'];
        if (!validTypes.includes(stock.type)) return false;
        uniqueSymbols.add(stock.symbol);
        return true;
      })
      .map(stock => ({
        symbol: stock.symbol,
        description: stock.description,
        type: stock.type
      }));

    console.log(`✓ Fetched ${stocks.length} stocks`);

    // Delete old cache
    await supabase.from('stock_cache').delete().eq('cache_key', 'all_stocks');

    // Insert new cache
    const { error } = await supabase
      .from('stock_cache')
      .insert({
        cache_key: 'all_stocks',
        stocks: stocks,
        updated_at: new Date().toISOString()
      });

    if (error) throw error;

    console.log(`✅ Successfully cached ${stocks.length} stocks in Supabase`);

    // Save backup
    await fs.mkdir('./data', { recursive: true });
    await fs.writeFile('./data/stock_cache.json', JSON.stringify(stocks, null, 2));
    console.log(`✓ Saved backup to data/stock_cache.json`);

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

fetchAndCacheStocks()
  .then(() => {
    console.log('\n✨ Stock cache updated successfully!');
    process.exit(0);
  })
  .catch(err => {
    console.error('💥 Fatal error:', err);
    process.exit(1);
  });