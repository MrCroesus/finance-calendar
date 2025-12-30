// FILE: api/calendar/[id].js
import { supabase } from '../../lib/supabase.js';
import yahooFinance from 'yahoo-finance2';
import { getFOMCDates, getBLSDates, getBEADates, formatEconomicEvent } from '../../lib/economic-calendar.js';

async function fetchEarningsDate(ticker) {
  try {
    console.log(`Attempting to fetch earnings for ${ticker}...`);
    
    // Fetch both calendar events and company name
    const result = await yahooFinance.quoteSummary(ticker, {
      modules: ['calendarEvents', 'price']
    });
    
    console.log(`Yahoo Finance response for ${ticker}:`, JSON.stringify(result, null, 2));
    
    const earnings = result.calendarEvents?.earnings;
    const companyName = result.price?.longName || result.price?.shortName || ticker;

    if (earnings?.earningsDate) {
      // Yahoo Finance can return multiple dates, take the first one
      const earningsDateValue = Array.isArray(earnings.earningsDate) 
        ? earnings.earningsDate[0] 
        : earnings.earningsDate;
      
      const earningsDate = new Date(earningsDateValue);
      
      console.log(`Found earnings date for ${ticker} (${companyName}): ${earningsDate.toISOString()}`);
      
      return {
        ticker,
        companyName,
        date: earningsDate,
        dateString: earningsDate.toISOString()
      };
    }

    console.warn(`No earnings date found for ${ticker}`);
    return null;
  } catch (error) {
    console.error(`Error fetching ${ticker}:`, error.message);
    
    // If ticker not found or other error, return null
    if (error.message.includes('Not Found') || error.message.includes('404')) {
      console.warn(`Ticker ${ticker} not found in Yahoo Finance`);
    }
    
    return null;
  }
}

async function getOrFetchTickerData(ticker) {
  // First, try to get from cache
  const { data: cachedTicker, error: fetchError } = await supabase
    .from('earnings_cache')
    .select('*')
    .eq('ticker', ticker)
    .single();

  let returnData = null;
  let shouldRefresh = false;

  if (cachedTicker && !fetchError && cachedTicker.earnings_date) {
    // We have cached data - use it immediately
    console.log(`Using cached data for ${ticker}`);
    returnData = {
      ticker: cachedTicker.ticker,
      companyName: cachedTicker.company_name || ticker,
      date: new Date(cachedTicker.earnings_date),
      dateString: cachedTicker.earnings_date
    };

    // Check if cache is old (older than 60 days)
    const lastUpdated = new Date(cachedTicker.last_updated);
    const now = new Date();
    const daysSinceUpdate = (now - lastUpdated) / (1000 * 60 * 60 * 24);
    
    if (daysSinceUpdate >= 60) {
      console.log(`Cache for ${ticker} is ${Math.round(daysSinceUpdate)} days old, will refresh in background`);
      shouldRefresh = true;
    }
  } else {
    // No cache or cache is null - fetch synchronously
    console.log(`No cached data for ${ticker}, fetching now...`);
    const freshData = await fetchEarningsDate(ticker);
    returnData = freshData;
    
    // Save to cache
    await supabase
      .from('earnings_cache')
      .upsert({
        ticker: ticker,
        company_name: freshData ? freshData.companyName : null,
        earnings_date: freshData ? freshData.dateString : null,
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });
  }

  // Trigger background refresh if needed (don't await)
  if (shouldRefresh) {
    refreshInBackground(ticker);
  }

  return returnData;
}

// Background refresh function - updates cache without blocking response
async function refreshInBackground(ticker) {
  try {
    console.log(`Background refresh started for ${ticker}`);
    const freshData = await fetchEarningsDate(ticker);
    
    await supabase
      .from('earnings_cache')
      .upsert({
        ticker: ticker,
        company_name: freshData ? freshData.companyName : null,
        earnings_date: freshData ? freshData.dateString : null,
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });
    
    console.log(`Background refresh completed for ${ticker}`);
  } catch (error) {
    console.error(`Background refresh failed for ${ticker}:`, error);
  }
}

function formatICSDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

function generateICS(earningsData, economicEvents, calendarId) {
  const now = new Date();
  const timestamp = formatICSDate(now) + 'T' + 
    now.toISOString().split('T')[1].replace(/[-:]/g, '').split('.')[0] + 'Z';

  const validEvents = earningsData.filter(item => item !== null);
  
  console.log(`Generating calendar with ${validEvents.length} earnings events and ${economicEvents.length} economic events`);

  // Generate earnings events
  const earningsEvents = validEvents.map(item => {
    const dateStr = formatICSDate(item.date);
    const uid = `earnings-${item.ticker}-${dateStr}-${calendarId}@earningscalendar.com`;

    return `BEGIN:VEVENT
UID:${uid}
DTSTAMP:${timestamp}
DTSTART;VALUE=DATE:${dateStr}
SUMMARY:${item.ticker} Earnings Report
DESCRIPTION:Earnings report for ${item.companyName}
STATUS:CONFIRMED
TRANSP:TRANSPARENT
END:VEVENT`;
  }).join('\n');

  // Generate economic events
  const economicEventsFormatted = economicEvents
    .map(event => formatEconomicEvent(event, calendarId))
    .join('\n');

  // Combine all events
  const allEvents = [earningsEvents, economicEventsFormatted].filter(Boolean).join('\n');

  return `BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Earnings Calendar Subscription//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
X-WR-CALNAME:Finance Calendar
X-WR-TIMEZONE:UTC
X-WR-CALDESC:Earnings dates for tracked stocks and economic calendar events
X-PUBLISHED-TTL:P60D
REFRESH-INTERVAL;VALUE=DURATION:P60D
${allEvents}
END:VCALENDAR`;
}

export default async function handler(req, res) {
  const { id } = req.query;

  if (!id) {
    return res.status(400).json({ error: 'Calendar ID required' });
  }

  try {
    console.log(`Fetching calendar ${id}`);
    
    const { data: calendar, error: fetchError } = await supabase
      .from('calendars')
      .select('*')
      .eq('id', id)
      .single();

    if (fetchError || !calendar) {
      console.error('Calendar not found:', fetchError);
      return res.status(404).json({ error: 'Calendar not found' });
    }

    console.log(`Calendar ${id} has ${calendar.tickers.length} tickers:`, calendar.tickers);

    // Fetch earnings data for tickers
    const earningsPromises = calendar.tickers.map(ticker => getOrFetchTickerData(ticker));
    const earningsData = await Promise.all(earningsPromises);

    // Fetch economic calendar events based on preferences
    const economicEvents = [];
    
    if (calendar.include_fomc) {
      console.log('Including FOMC dates');
      const fomcDates = await getFOMCDates();
      economicEvents.push(...fomcDates);
    }
    
    if (calendar.include_bls) {
      console.log('Including BLS dates');
      const blsDates = getBLSDates();
      economicEvents.push(...blsDates);
    }
    
    if (calendar.include_bea) {
      console.log('Including BEA dates');
      const beaDates = getBEADates();
      economicEvents.push(...beaDates);
    }

    const icsContent = generateICS(earningsData, economicEvents, id);

    res.setHeader('Content-Type', 'text/calendar; charset=utf-8');
    res.setHeader('Content-Disposition', `inline; filename="earnings-calendar.ics"`);
    res.setHeader('Cache-Control', 'public, max-age=5184000');

    res.status(200).send(icsContent);
  } catch (error) {
    console.error('Error generating calendar:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}