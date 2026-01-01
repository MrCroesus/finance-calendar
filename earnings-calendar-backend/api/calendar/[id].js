// api/calendar/[id].js
// Generates .ics calendar feed for a given calendar ID

import { supabase } from '../../lib/supabase.js';
import { getEconomicEvents, eventToICS } from '../../lib/economic-calendar.js';
import yahooFinance from 'yahoo-finance2';

export default async function handler(req, res) {
  const { id } = req.query;

  try {
    console.log(`📅 Generating calendar for ID: ${id}`);

    // ============================================================
    // Step 1: Get calendar preferences from database
    // ============================================================
    
    const { data: calendar, error: calendarError } = await supabase
      .from('calendars')
      .select('*')
      .eq('id', id)
      .single();

    if (calendarError || !calendar) {
      console.error('Calendar not found:', id);
      return res.status(404).json({ error: 'Calendar not found' });
    }

    console.log(`✓ Found calendar with ${calendar.tickers?.length || 0} tickers`);

    // ============================================================
    // Step 2: Fetch earnings dates for all tickers
    // ============================================================
    
    const earningsEvents = [];
    const tickers = calendar.tickers || [];

    for (const ticker of tickers) {
      try {
        const earningsDate = await getEarningsDate(ticker);
        
        if (earningsDate) {
          earningsEvents.push({
            dtstart: {
              date: earningsDate.date,
              time: earningsDate.time || '13:00:00', // Default to 1 PM UTC if no time
              isAllDay: !earningsDate.time,
              timestamp: earningsDate.timestamp
            },
            summary: `${ticker} Earnings`,
            description: `${earningsDate.companyName || ticker} earnings release`,
            uid: `earnings-${ticker}-${earningsDate.date}@${id}`
          });
        }
      } catch (err) {
        console.error(`Error fetching earnings for ${ticker}:`, err.message);
        // Continue with other tickers even if one fails
      }
    }

    console.log(`✓ Found ${earningsEvents.length} earnings events`);

    // ============================================================
    // Step 3: Fetch economic calendar events
    // ============================================================
    
    const economicEvents = await getEconomicEvents(
      calendar.include_fomc,
      calendar.include_bls,
      calendar.include_bea
    );

    console.log(`✓ Found ${economicEvents.length} economic events`);

    // ============================================================
    // Step 4: Combine all events
    // ============================================================
    
    const allEvents = [...earningsEvents, ...economicEvents];

    // Sort by date
    allEvents.sort((a, b) => {
      const dateA = a.dtstart.timestamp || a.dtstart.date;
      const dateB = b.dtstart.timestamp || b.dtstart.date;
      return new Date(dateA) - new Date(dateB);
    });

    console.log(`✓ Total events: ${allEvents.length}`);

    // ============================================================
    // Step 5: Generate ICS file
    // ============================================================
    
    const icsEvents = allEvents.map(event => eventToICS(event, id));

    const icsContent = [
      'BEGIN:VCALENDAR',
      'VERSION:2.0',
      'PRODID:-//Earnings Calendar//Finance Events//EN',
      'CALSCALE:GREGORIAN',
      'METHOD:PUBLISH',
      'X-WR-CALNAME:Finance Calendar',
      'X-WR-CALDESC:Stock earnings and economic events',
      'X-WR-TIMEZONE:UTC',
      ...icsEvents,
      'END:VCALENDAR'
    ].join('\r\n');

    // ============================================================
    // Step 6: Return ICS file
    // ============================================================
    
    res.setHeader('Content-Type', 'text/calendar; charset=utf-8');
    res.setHeader('Content-Disposition', `inline; filename="calendar-${id}.ics"`);
    res.setHeader('Cache-Control', 'public, max-age=3600'); // Cache for 1 hour
    
    return res.status(200).send(icsContent);

  } catch (error) {
    console.error('Error generating calendar:', error);
    return res.status(500).json({ 
      error: 'Failed to generate calendar',
      message: error.message 
    });
  }
}

// ============================================================
// Helper: Get earnings date for a ticker
// ============================================================

async function getEarningsDate(ticker) {
  try {
    // Check cache first
    const cached = await getEarningsFromCache(ticker);
    
    if (cached) {
      const cacheAge = Date.now() - new Date(cached.last_updated).getTime();
      const SIXTY_DAYS = 60 * 24 * 60 * 60 * 1000;
      
      // If cache is less than 60 days old, use it
      if (cacheAge < SIXTY_DAYS) {
        console.log(`  ✓ ${ticker}: Using cached earnings date`);
        return {
          date: cached.earnings_date.split('T')[0],
          timestamp: cached.earnings_date,
          companyName: cached.company_name
        };
      }
      
      // If cache is old, serve it but refresh in background
      console.log(`  ⚠️  ${ticker}: Cache expired, refreshing...`);
      refreshEarningsCache(ticker).catch(err => {
        console.error(`Background refresh failed for ${ticker}:`, err);
      });
      
      return {
        date: cached.earnings_date.split('T')[0],
        timestamp: cached.earnings_date,
        companyName: cached.company_name
      };
    }

    // Not in cache, fetch from Yahoo Finance
    console.log(`  📡 ${ticker}: Fetching from Yahoo Finance...`);
    return await fetchAndCacheEarnings(ticker);

  } catch (error) {
    console.error(`Error getting earnings for ${ticker}:`, error);
    return null;
  }
}

// ============================================================
// Get earnings from cache
// ============================================================

async function getEarningsFromCache(ticker) {
  try {
    const { data, error } = await supabase
      .from('earnings_cache')
      .select('*')
      .eq('ticker', ticker.toUpperCase())
      .single();

    if (error) {
      // Not found is expected, not an error
      if (error.code === 'PGRST116') return null;
      throw error;
    }

    return data;
  } catch (error) {
    console.error(`Cache lookup error for ${ticker}:`, error);
    return null;
  }
}

// ============================================================
// Fetch earnings from Yahoo Finance and cache
// ============================================================

async function fetchAndCacheEarnings(ticker) {
  try {
    // Fetch quote data which includes earnings date
    const quote = await yahooFinance.quote(ticker);
    
    if (!quote) {
      throw new Error('No quote data returned');
    }

    const earningsDate = quote.earningsTimestamp 
      ? new Date(quote.earningsTimestamp * 1000)
      : quote.earningsDate 
        ? new Date(quote.earningsDate)
        : null;

    if (!earningsDate || isNaN(earningsDate.getTime())) {
      console.warn(`  ⚠️  ${ticker}: No earnings date available`);
      return null;
    }

    const companyName = quote.longName || quote.shortName || ticker;

    // Cache the result
    const { error } = await supabase
      .from('earnings_cache')
      .upsert({
        ticker: ticker.toUpperCase(),
        company_name: companyName,
        earnings_date: earningsDate.toISOString(),
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });

    if (error) {
      console.error(`Failed to cache ${ticker}:`, error);
    } else {
      console.log(`  ✓ ${ticker}: Cached earnings date`);
    }

    return {
      date: earningsDate.toISOString().split('T')[0],
      timestamp: earningsDate.toISOString(),
      companyName: companyName
    };

  } catch (error) {
    console.error(`Yahoo Finance fetch failed for ${ticker}:`, error);
    return null;
  }
}

// ============================================================
// Refresh cache in background (fire and forget)
// ============================================================

async function refreshEarningsCache(ticker) {
  try {
    const quote = await yahooFinance.quote(ticker);
    
    if (!quote) return;

    const earningsDate = quote.earningsTimestamp 
      ? new Date(quote.earningsTimestamp * 1000)
      : quote.earningsDate 
        ? new Date(quote.earningsDate)
        : null;

    if (!earningsDate || isNaN(earningsDate.getTime())) {
      return;
    }

    const companyName = quote.longName || quote.shortName || ticker;

    await supabase
      .from('earnings_cache')
      .upsert({
        ticker: ticker.toUpperCase(),
        company_name: companyName,
        earnings_date: earningsDate.toISOString(),
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });

    console.log(`  ✓ ${ticker}: Background refresh complete`);
  } catch (error) {
    console.error(`Background refresh error for ${ticker}:`, error);
  }
}