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

  // DEBUG MODE - add ?debug=true to URL
  const debugMode = req.query.debug === 'true';
  const debugLogs = [];
  const debugLog = (msg) => {
    console.log(msg);
    if (debugMode) debugLogs.push(msg);
  };

  // Handle PUT request to update calendar
  if (req.method === 'PUT') {
    return handleUpdateCalendar(req, res, id);
  }

  // Handle GET request to generate ICS
  if (req.method === 'GET') {
    return handleGenerateICS(req, res, id, debugLog, debugMode, debugLogs);
  }

  return res.status(405).json({ error: 'Method not allowed' });
}

// Update calendar preferences
async function handleUpdateCalendar(req, res, id) {
  try {
    const { tickers, include_fomc, include_bls, include_bea } = req.body;

    if (!Array.isArray(tickers)) {
      return res.status(400).json({ error: 'tickers must be an array' });
    }

    const { data, error } = await supabase
      .from('calendars')
      .update({
        tickers: tickers,
        include_fomc: include_fomc ?? false,
        include_bls: include_bls ?? false,
        include_bea: include_bea ?? false
      })
      .eq('id', id)
      .select()
      .single();

    if (error || !data) {
      console.error('Update error:', error);
      return res.status(404).json({ error: 'Calendar not found' });
    }

    return res.status(200).json({
      message: 'Calendar updated successfully',
      calendar: data
    });
  } catch (error) {
    console.error('Error updating calendar:', error);
    return res.status(500).json({ 
      error: 'Failed to update calendar',
      message: error.message 
    });
  }
}

// Generate ICS file
async function handleGenerateICS(req, res, id, debugLog, debugMode, debugLogs) {
  try {
    debugLog(`📅 Generating calendar for ID: ${id}`);

    const { data: calendar, error: calendarError } = await supabase
      .from('calendars')
      .select('*')
      .eq('id', id)
      .single();

    if (calendarError || !calendar) {
      debugLog(`❌ Calendar not found: ${id}`);
      if (debugMode) {
        return res.status(404).json({ error: 'Calendar not found', debug: debugLogs });
      }
      return res.status(404).json({ error: 'Calendar not found' });
    }

    debugLog(`✓ Found calendar with ${calendar.tickers?.length || 0} tickers`);

    const earningsEvents = [];
    const tickers = calendar.tickers || [];

    for (const ticker of tickers) {
      try {
        debugLog(`📊 Fetching earnings for ${ticker}...`);
        const earningsDate = await getEarningsDate(ticker, debugLog);
        
        if (earningsDate) {
          debugLog(`  ✓ Found earnings for ${ticker}: ${earningsDate.date}, timing: ${earningsDate.timing || 'unknown'}`);
          
          let eventTime, eventDescription;
          
          if (earningsDate.timing === 'BMO') {
            eventTime = '12:00:00';
            eventDescription = `${earningsDate.companyName || ticker} earnings release (Before Market Open)`;
          } else if (earningsDate.timing === 'AMC') {
            eventTime = '21:00:00';
            eventDescription = `${earningsDate.companyName || ticker} earnings release (After Market Close)`;
          } else {
            earningsEvents.push({
              dtstart: {
                date: earningsDate.date,
                isAllDay: true
              },
              summary: `${ticker} Earnings`,
              description: `${earningsDate.companyName || ticker} earnings release`,
              uid: `earnings-${ticker}-${earningsDate.date}@${id}`
            });
            debugLog(`  → Added as all-day event (no timing info)`);
            continue;
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
          debugLog(`  → Added as timed event (${earningsDate.timing})`);
        } else {
          debugLog(`  ⚠️  No earnings date found for ${ticker}`);
        }
      } catch (err) {
        debugLog(`❌ Error fetching earnings for ${ticker}: ${err.message}`);
      }
    }

    debugLog(`✓ Found ${earningsEvents.length} earnings events`);

    const economicEvents = await getEconomicEvents(
      calendar.include_fomc,
      calendar.include_bls,
      calendar.include_bea
    );

    debugLog(`✓ Found ${economicEvents.length} economic events`);

    const allEvents = [...earningsEvents, ...economicEvents];

    allEvents.sort((a, b) => {
      const dateA = a.dtstart.timestamp || a.dtstart.date;
      const dateB = b.dtstart.timestamp || b.dtstart.date;
      return new Date(dateA) - new Date(dateB);
    });

    debugLog(`✓ Total events: ${allEvents.length}`);

    // If debug mode, return JSON instead of ICS
    if (debugMode) {
      return res.status(200).json({
        debug: debugLogs,
        calendarInfo: {
          id: calendar.id,
          tickers: calendar.tickers,
          earningsEventsCount: earningsEvents.length,
          economicEventsCount: economicEvents.length,
          totalEvents: allEvents.length
        }
      });
    }

    const icsEvents = allEvents
      .map(event => eventToICS(event, id))
      .filter(ics => ics !== '');

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
    res.setHeader('Content-Disposition', `inline; filename="finance-calendar.ics"`);
    res.setHeader('Cache-Control', 'public, max-age=3600');
    
    return res.status(200).send(icsContent);

  } catch (error) {
    console.error('Error generating calendar:', error);
    if (debugMode) {
      return res.status(500).json({ 
        error: 'Failed to generate calendar',
        message: error.message,
        debug: debugLogs
      });
    }
    return res.status(500).json({ 
      error: 'Failed to generate calendar',
      message: error.message 
    });
  }
}

async function getEarningsDate(ticker, debugLog = console.log) {
  try {
    const cached = await getEarningsFromCache(ticker);
    
    if (cached) {
      const cacheAge = Date.now() - new Date(cached.last_updated).getTime();
      const SIXTY_DAYS = 60 * 24 * 60 * 60 * 1000;
      
      debugLog(`  📦 Cache for ${ticker}: ${Math.round(cacheAge / 1000 / 60 / 60 / 24)} days old (limit: 60 days)`);
      
      if (cacheAge < SIXTY_DAYS) {
        debugLog(`  ✓ ${ticker}: Using cached earnings date`);
        return {
          date: cached.earnings_date.split('T')[0],
          timestamp: cached.earnings_date,
          companyName: cached.company_name,
          timing: cached.earnings_timing || null
        };
      }
      
      debugLog(`  ⚠️  ${ticker}: Cache expired (${Math.round(cacheAge / 1000 / 60 / 60 / 24)} days), fetching fresh...`);
      
      // CHANGED: Don't do background refresh, fetch synchronously
      const fresh = await fetchAndCacheEarnings(ticker, debugLog);
      if (fresh) {
        return fresh;
      }
      
      // If fetch failed, return stale cache
      debugLog(`  ⚠️  ${ticker}: Fresh fetch failed, using stale cache`);
      return {
        date: cached.earnings_date.split('T')[0],
        timestamp: cached.earnings_date,
        companyName: cached.company_name,
        timing: cached.earnings_timing || null
      };
    }

    debugLog(`  📡 ${ticker}: No cache, fetching from Yahoo Finance...`);
    return await fetchAndCacheEarnings(ticker, debugLog);

  } catch (error) {
    debugLog(`❌ Error getting earnings for ${ticker}: ${error.message}`);
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

async function fetchAndCacheEarnings(ticker, debugLog = console.log) {
  try {
    debugLog(`    🔍 Calling Yahoo Finance quoteSummary for ${ticker}...`);
    
    const data = await yahooFinance.quoteSummary(ticker, {
      modules: ['calendarEvents', 'price']
    });
    
    if (!data || !data.calendarEvents) {
      debugLog(`    ❌ ${ticker}: No calendar events data from Yahoo`);
      return null;
    }

    const earnings = data.calendarEvents.earnings;
    
    if (!earnings || !earnings.earningsDate || earnings.earningsDate.length === 0) {
      debugLog(`    ❌ ${ticker}: No earnings date in calendar events`);
      return null;
    }

    const earningsDate = earnings.earningsDate[0];
    
    if (!earningsDate || 
        isNaN(earningsDate.getTime()) || 
        earningsDate.getFullYear() < 2000 || 
        earningsDate.getFullYear() > 2100) {
      debugLog(`    ❌ ${ticker}: Invalid earnings date (year: ${earningsDate?.getFullYear()})`);
      return null;
    }

    const companyName = data.price?.longName || data.price?.shortName || ticker;
    
    let timing = null;
    const hour = earningsDate.getUTCHours();
    
    if (hour >= 4 && hour < 14) {
      timing = 'BMO';
    } else if (hour >= 20 || hour < 4) {
      timing = 'AMC';
    }

    debugLog(`    💾 Saving to Supabase: ${ticker}, date: ${earningsDate.toISOString()}, timing: ${timing}`);

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
      debugLog(`    ❌ Failed to cache ${ticker} in Supabase: ${error.message}`);
      console.error(`Failed to cache ${ticker}:`, error);
    } else {
      debugLog(`    ✅ ${ticker}: Successfully cached in Supabase`);
    }

    return {
      date: earningsDate.toISOString().split('T')[0],
      timestamp: earningsDate.toISOString(),
      companyName: companyName,
      timing: timing
    };

  } catch (error) {
    debugLog(`    ❌ Yahoo Finance fetch failed for ${ticker}: ${error.message}`);
    console.error(`Yahoo Finance fetch failed for ${ticker}:`, error.message);
    return null;
  }
}