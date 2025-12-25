import { supabase } from '../../lib/supabase.js';
import fetch from 'node-fetch';

async function fetchEarningsDate(ticker) {
  try {
    const response = await fetch(
      `https://query2.finance.yahoo.com/v10/finance/quoteSummary/${ticker}?modules=calendarEvents`,
      {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      }
    );

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    const earnings = data.quoteSummary?.result?.[0]?.calendarEvents?.earnings;

    if (earnings?.earningsDate?.[0]) {
      const earningsDate = new Date(earnings.earningsDate[0].raw * 1000);
      return {
        ticker,
        date: earningsDate,
        dateString: earningsDate.toISOString()
      };
    }

    return null;
  } catch (error) {
    console.error(`Error fetching ${ticker}:`, error);
    return null;
  }
}

async function getOrFetchTickerData(ticker) {
  // Try to get from cache first
  const { data: cachedTicker, error: fetchError } = await supabase
    .from('ticker_cache')
    .select('*')
    .eq('ticker', ticker)
    .single();

  // If we have cached data and it's less than 60 days old, use it
  if (cachedTicker && !fetchError) {
    const lastUpdated = new Date(cachedTicker.last_updated);
    const now = new Date();
    const daysSinceUpdate = (now - lastUpdated) / (1000 * 60 * 60 * 24);

    if (daysSinceUpdate < 60) {
      console.log(`Using cached data for ${ticker} (${Math.round(daysSinceUpdate)} days old)`);
      return cachedTicker.earnings_date ? {
        ticker: cachedTicker.ticker,
        date: new Date(cachedTicker.earnings_date),
        dateString: cachedTicker.earnings_date
      } : null;
    }
  }

  // Fetch fresh data from Yahoo Finance
  console.log(`Fetching fresh data for ${ticker}`);
  const freshData = await fetchEarningsDate(ticker);

  // Update cache
  if (freshData) {
    await supabase
      .from('ticker_cache')
      .upsert({
        ticker: ticker,
        earnings_date: freshData.dateString,
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });
  } else {
    // Even if no earnings date found, update the timestamp so we don't keep trying
    await supabase
      .from('ticker_cache')
      .upsert({
        ticker: ticker,
        earnings_date: null,
        last_updated: new Date().toISOString()
      }, {
        onConflict: 'ticker'
      });
  }

  return freshData;
}

function formatICSDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

function generateICS(earningsData, calendarId) {
  const now = new Date();
  const timestamp = formatICSDate(now) + 'T' + 
    now.toISOString().split('T')[1].replace(/[-:]/g, '').split('.')[0] + 'Z';

  const events = earningsData
    .filter(item => item !== null)
    .map(item => {
      const dateStr = formatICSDate(item.date);
      const uid = `earnings-${item.ticker}-${dateStr}-${calendarId}@earningscalendar.com`;

      return `BEGIN:VEVENT
UID:${uid}
DTSTAMP:${timestamp}
DTSTART;VALUE=DATE:${dateStr}
SUMMARY:${item.ticker} Earnings Report
DESCRIPTION:Earnings report for ${item.ticker}
STATUS:CONFIRMED
TRANSP:TRANSPARENT
END:VEVENT`;
    }).join('\n');

  return `BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Earnings Calendar Subscription//EN
CALSCALE:GREGORIAN
METHOD:PUBLISH
X-WR-CALNAME:Stock Earnings Calendar
X-WR-TIMEZONE:UTC
X-WR-CALDESC:Earnings dates for tracked stocks
X-PUBLISHED-TTL:P60D
REFRESH-INTERVAL;VALUE=DURATION:P60D
${events}
END:VCALENDAR`;
}

export default async function handler(req, res) {
  const { id } = req.query;

  if (!id) {
    return res.status(400).json({ error: 'Calendar ID required' });
  }

  try {
    const { data: calendar, error: fetchError } = await supabase
      .from('calendars')
      .select('*')
      .eq('id', id)
      .single();

    if (fetchError || !calendar) {
      return res.status(404).json({ error: 'Calendar not found' });
    }

    // Fetch earnings data for all tickers (uses cache when available)
    const earningsPromises = calendar.tickers.map(ticker => getOrFetchTickerData(ticker));
    const earningsData = await Promise.all(earningsPromises);

    // Generate ICS file
    const icsContent = generateICS(earningsData, id);

    res.setHeader('Content-Type', 'text/calendar; charset=utf-8');
    res.setHeader('Content-Disposition', `inline; filename="earnings-calendar.ics"`);
    res.setHeader('Cache-Control', 'public, max-age=5184000');

    res.status(200).send(icsContent);
  } catch (error) {
    console.error('Error generating calendar:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}