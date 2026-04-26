// scripts/refresh-watched-earnings.js
import { createClient } from '@supabase/supabase-js';
import yahooFinance from 'yahoo-finance2';
import fs from 'fs/promises';

// Load env
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

async function refreshWatchedEarnings() {
  console.log('🔄 Refreshing watched earnings...\n');
  
  try {
    // Get all calendars
    const { data: calendars, error: calendarsError } = await supabase
      .from('calendars')
      .select('tickers');

    if (calendarsError) {
      throw calendarsError;
    }

    // Get union of all tickers across all calendars
    const allTickers = new Set();
    calendars.forEach(calendar => {
      if (calendar.tickers && Array.isArray(calendar.tickers)) {
        calendar.tickers.forEach(ticker => allTickers.add(ticker.toUpperCase()));
      }
    });

    const uniqueTickers = Array.from(allTickers).sort();
    console.log(`📊 Found ${uniqueTickers.length} unique tickers across ${calendars.length} calendars`);
    console.log(`Tickers: ${uniqueTickers.join(', ')}\n`);

    let refreshed = 0;
    let failed = 0;
    let skipped = 0;

    for (const ticker of uniqueTickers) {
      try {
        console.log(`📊 Fetching ${ticker}...`);
        
        const data = await yahooFinance.quoteSummary(ticker, {
          modules: ['calendarEvents', 'price']
        });
        
        if (!data?.calendarEvents?.earnings?.earningsDate?.[0]) {
          console.log(`  ⚠️  No earnings date for ${ticker}`);
          skipped++;
          continue;
        }

        const earningsDate = data.calendarEvents.earnings.earningsDate[0];
        
        if (isNaN(earningsDate.getTime()) || 
            earningsDate.getFullYear() < 2000 || 
            earningsDate.getFullYear() > 2100) {
          console.log(`  ⚠️  Invalid date for ${ticker}`);
          skipped++;
          continue;
        }

        const companyName = data.price?.longName || data.price?.shortName || ticker;
        const hour = earningsDate.getUTCHours();
        let timing = null;
        
        if (hour >= 4 && hour < 14) timing = 'BMO';
        else if (hour >= 20 || hour < 4) timing = 'AMC';

        const { error } = await supabase
          .from('earnings_cache')
          .upsert({
            ticker: ticker.toUpperCase(),
            company_name: companyName,
            earnings_date: earningsDate.toISOString(),
            earnings_timing: timing,
            last_updated: new Date().toISOString()
          }, {
            onConflict: 'ticker'
          });

        if (error) {
          console.log(`  ❌ Failed to cache ${ticker}: ${error.message}`);
          failed++;
        } else {
          console.log(`  ✅ ${ticker}: ${earningsDate.toISOString().split('T')[0]} (${timing || 'unknown'})`);
          refreshed++;
        }

        // Rate limit: wait 2 seconds between requests to avoid "Too Many Requests"
        await new Promise(resolve => setTimeout(resolve, 2000));

      } catch (error) {
        console.log(`  ❌ Error fetching ${ticker}:`);
        console.log(`     Error type: ${error.constructor.name}`);
        console.log(`     Error message: ${error.message}`);
        
        // If there's a response, show it
        if (error.result) {
          console.log(`     Yahoo response:`, JSON.stringify(error.result).substring(0, 200));
        }
        
        failed++;
        
        // Only wait if it's actually a rate limit
        if (error.message.includes('Too Many Requests') || 
            error.message.includes('429') ||
            error.message.includes('Rate limit')) {
          console.log('  ⏳ Rate limited, waiting 30 seconds...');
          await new Promise(resolve => setTimeout(resolve, 30000));
        } else {
          // For other errors, just wait 2 seconds and continue
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }
    }

    console.log(`\n✨ Refresh complete:`);
    console.log(`   ✅ ${refreshed} refreshed`);
    console.log(`   ❌ ${failed} failed`);
    console.log(`   ⏭️  ${skipped} skipped (no data)`);
    console.log(`   📊 ${uniqueTickers.length} total tickers`);

  } catch (error) {
    console.error('💥 Fatal error:', error.message);
    process.exit(1);
  }
}

refreshWatchedEarnings()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('💥 Fatal error:', err);
    process.exit(1);
  });