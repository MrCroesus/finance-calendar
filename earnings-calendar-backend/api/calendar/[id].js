// api/calendar/[id].js
import { supabase } from '../../lib/supabase.js';
import { getEconomicEvents, eventToICS } from '../../lib/economic-calendar.js';
import yahooFinance from 'yahoo-finance2';

export default async function handler(req, res) {
  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.status(200).end();
  }

  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  const { id } = req.query;

  try {
    console.log(`📅 Generating calendar for ID: ${id}`);

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

    const earningsEvents = [];
    const tickers = calendar.tickers || [];

    for (const ticker of tickers) {
      try {
        const earningsDate = await getEarningsDate(ticker);
        
        if (earningsDate) {
          let eventTime, eventDescription;
          
          if (earningsDate.timing === 'BMO') {
            eventTime = '12:00:00';
            eventDescription = `${earningsDate.companyName || ticker} earnings release (Before Market Open)`;
          } else if (earningsDate.timing === 'AMC') {
            eventTime = '21:00:00';
            eventDescription = `${earningsDate.companyName || ticker} earnings release (After Market Close)`;
          } else {
            eventTime = '13:00:00';
            eventDescription = `${earningsDate.companyName || ticker} earnings release`;
          }

          earningsEvents.push({
            dtstart: {
              date: earningsDate.date,
              time: eventTime,
              isAllDay: false,
              timestamp: `${earningsDate.date}T${eventTime}Z`
            },
            summary: `${ticker} Earnings`,
            description: eventDescription,
            uid: `earnings-${ticker}-${earningsDate.date}@${id}`
          });
        }
      } catch (err) {
        console.error(`Error fetching earnings for ${ticker}:`, err.message);
      }
    }

    console.log(`✓ Found ${earningsEvents.length} earnings events`);

    const economicEvents = await getEconomicEvents(
      calendar.include_fomc,
      calendar.include_bls,
      calendar.include_bea
    );

    console.log(`✓ Found ${economicEvents.length} economic events`);

    const allEvents = [...earningsEvents, ...economicEvents];

    allEvents.sort((a, b) => {
      const dateA = a.dtstart.timestamp || a.dtstart.date;
      const dateB = b.dtstart.timestamp || b.dtstart.date;
      return new Date(dateA) - new Date(dateB);
    });

    console.log(`✓ Total events: ${allEvents.length}`);

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

    res.setHeader('Content-Type', 'text/calendar; charset=utf-8');
    res.setHeader('Content-Disposition', `inline; filename="calendar-${id}.ics"`);
    res.setHeader('Cache-Control', 'public, max-age=3600');
    
    return res.status(200).send(icsContent);

  } catch (error) {
    console.error('Error generating calendar:', error);
    return res.status(500).json({ 
      error: 'Failed to generate calendar',
      message: error.message 
    });
  }
}

async function getEarningsDate(ticker) {
  try {
    const cached = await getEarningsFromCache(ticker);
    
    if (cached) {
      const cacheAge = Date.now() - new Date(cached.last_updated).getTime();
      const SIXTY_DAYS = 60 * 24 * 60 * 60 * 1000;
      
      if (cacheAge < SIXTY_DAYS) {
        console.log(`  ✓ ${ticker}: Using cached earnings date`);
        return {
          date: cached.earnings_date.split('T')[0],
          timestamp: cached.earnings_date,
          companyName: cached.company_name,
          timing: cached.earnings_timing || null
        };
      }
      
      console.log(`  ⚠️  ${ticker}: Cache expired, refreshing...`);
      refreshEarningsCache(ticker).catch(err => {
        console.error(`Background refresh failed for ${ticker}:`, err);
      });
      
      return {
        date: cached.earnings_date.split('T')[0],
        timestamp: cached.earnings_date,
        companyName: cached.company_name,
        timing: cached.earnings_timing || null
      };
    }

    console.log(`  📡 ${ticker}: Fetching from Yahoo Finance...`);
    return await fetchAndCacheEarnings(ticker);

  } catch (error) {
    console.error(`Error getting earnings for ${ticker}:`, error);
    return null;
  }
}

async function getEarningsFromCache(ticker) {
  try {
    const { data, error } = await supabase
      .from('earnings_cache')
      .select('*')
      .eq('ticker', ticker.toUpperCase())
      .single();

    if (error) {
      if (error.code === 'PGRST116') return null;
      throw error;
    }

    return data;
  } catch (error) {
    console.error(`Cache lookup error for ${ticker}:`, error);
    return null;
  }
}

async function fetchAndCacheEarnings(ticker) {
  try {
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
    
    let timing = null;
    const hour = earningsDate.getUTCHours();
    
    if (hour >= 4 && hour < 14) {
      timing = 'BMO';
    } else if (hour >= 20 || hour < 4) {
      timing = 'AMC';
    }

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
      console.error(`Failed to cache ${ticker}:`, error);
    } else {
      console.log(`  ✓ ${ticker}: Cached earnings date (${timing || 'unknown timing'})`);
    }

    return {
      date: earningsDate.toISOString().split('T')[0],
      timestamp: earningsDate.toISOString(),
      companyName: companyName,
      timing: timing
    };

  } catch (error) {
    console.error(`Yahoo Finance fetch failed for ${ticker}:`, error);
    return null;
  }
}

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
    
    let timing = null;
    const hour = earningsDate.getUTCHours();
    
    if (hour >= 4 && hour < 14) {
      timing = 'BMO';
    } else if (hour >= 20 || hour < 4) {
      timing = 'AMC';
    }

    await supabase
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

    console.log(`  ✓ ${ticker}: Background refresh complete`);
  } catch (error) {
    console.error(`Background refresh error for ${ticker}:`, error);
  }
}