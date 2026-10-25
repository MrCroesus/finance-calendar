// lib/refresh.js
import YahooFinance from 'yahoo-finance2';
import { supabase } from './supabase.js';

const yahooFinance = new YahooFinance();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------
// Stocks (Finnhub -> stock_cache)
// ---------------------------------------------------------------
export async function refreshStocks() {
  const key = process.env.FINNHUB_API_KEY;
  if (!key) throw new Error('FINNHUB_API_KEY not set');

  const exchanges = ['US', 'OTC'];
  const settled = await Promise.allSettled(
    exchanges.map(async (exchange) => {
      const res = await fetch(
        `https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${key}`
      );
      if (!res.ok) throw new Error(`Finnhub ${exchange}: HTTP ${res.status}`);
      return res.json();
    })
  );

  const failures = settled
    .filter((r) => r.status === 'rejected')
    .map((r) => r.reason.message);
  if (failures.length === exchanges.length) throw new Error(failures.join('; '));

  const validTypes = ['Common Stock', 'ADR', 'GDR'];
  const merged = new Map();

  // Partial failure: keep previously cached stocks so the failed exchange isn't wiped
  if (failures.length > 0) {
    const { data: existing } = await supabase
      .from('stock_cache')
      .select('stocks')
      .eq('cache_key', 'all_stocks')
      .single();
    (existing?.stocks || []).forEach((s) => merged.set(s.symbol, s));
  }

  settled
    .filter((r) => r.status === 'fulfilled')
    .forEach((r) =>
      r.value.forEach((s) => {
        if (!s.symbol || !s.description || !validTypes.includes(s.type)) return;
        merged.set(s.symbol, { symbol: s.symbol, description: s.description, type: s.type });
      })
    );

  const stocks = [...merged.values()];
  if (stocks.length === 0) throw new Error('No stocks to cache');

  await supabase.from('stock_cache').delete().eq('cache_key', 'all_stocks');
  const { error } = await supabase.from('stock_cache').insert({
    cache_key: 'all_stocks',
    stocks,
    updated_at: new Date().toISOString(),
  });
  if (error) throw error;

  return { count: stocks.length, warnings: failures };
}

// ---------------------------------------------------------------
// BLS / BEA (ICS -> economic_calendar_cache)
// ---------------------------------------------------------------
const ECON_SOURCES = [
  { name: 'BLS', url: 'https://www.bls.gov/schedule/news_release/bls.ics', cacheKey: 'bls_calendar' },
  { name: 'BEA', url: 'https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics', cacheKey: 'bea_calendar' },
];

export async function refreshEconomicCalendars() {
  const out = {};
  for (const src of ECON_SOURCES) {
    try {
      const res = await fetch(src.url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; EarningsCalendar/1.0)',
          Accept: 'text/calendar,*/*',
        },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const events = parseICS(await res.text());
      if (events.length === 0) throw new Error('Parsed 0 events');

      const { error } = await supabase.from('economic_calendar_cache').upsert(
        { cache_key: src.cacheKey, events, last_updated: new Date().toISOString() },
        { onConflict: 'cache_key' }
      );
      if (error) throw error;
      out[src.name] = { ok: true, count: events.length };
    } catch (err) {
      // Old cache stays in place if this fails
      out[src.name] = { ok: false, error: err.message };
    }
  }
  return out;
}

// ---------------------------------------------------------------
// Earnings (Yahoo -> earnings_cache), stalest first, with time budget
// ---------------------------------------------------------------
export async function refreshEarnings(deadline) {
  const { data: calendars, error } = await supabase.from('calendars').select('tickers');
  if (error) throw error;

  const tickers = new Set();
  calendars.forEach((c) => (c.tickers || []).forEach((t) => tickers.add(t.toUpperCase())));
  const all = [...tickers];

  // Order by last_updated ascending (never-cached tickers first)
  const { data: cached } = await supabase
    .from('earnings_cache')
    .select('ticker, last_updated')
    .in('ticker', all);
  const updatedAt = new Map((cached || []).map((r) => [r.ticker, new Date(r.last_updated).getTime()]));
  all.sort((a, b) => (updatedAt.get(a) || 0) - (updatedAt.get(b) || 0));

  const summary = { total: all.length, refreshed: 0, failed: 0, skipped: 0, notReached: 0, errors: [] };

  for (let i = 0; i < all.length; i++) {
    if (Date.now() > deadline) {
      summary.notReached = all.length - i;
      break;
    }
    const ticker = all[i];
    try {
      const data = await yahooFinance.quoteSummary(ticker, { modules: ['calendarEvents', 'price'] });
      const date = data?.calendarEvents?.earnings?.earningsDate?.[0];

      if (!date || isNaN(date.getTime()) || date.getFullYear() < 2000 || date.getFullYear() > 2100) {
        summary.skipped++;
      } else {
        const hour = date.getUTCHours();
        let timing = null;
        if (hour >= 4 && hour < 14) timing = 'BMO';
        else if (hour >= 20 || hour < 4) timing = 'AMC';

        const { error: upErr } = await supabase.from('earnings_cache').upsert(
          {
            ticker,
            company_name: data.price?.longName || data.price?.shortName || ticker,
            earnings_date: date.toISOString(),
            earnings_timing: timing,
            last_updated: new Date().toISOString(),
          },
          { onConflict: 'ticker' }
        );
        if (upErr) throw upErr;
        summary.refreshed++;
      }
    } catch (err) {
      summary.failed++;
      summary.errors.push(`${ticker}: ${err.message}`);
    }
    await sleep(2000);
  }
  return summary;
}

// ---------------------------------------------------------------
// ICS parser
// ---------------------------------------------------------------
function parseICS(ics) {
  const events = [];
  const lines = ics.split(/\r?\n/);
  let current = null;
  let field = null;
  let value = '';

  for (const raw of lines) {
    if (/^[ \t]/.test(raw) && field) {
      value += raw.substring(1);
      continue;
    }
    if (field && current) processField(current, field, value);
    field = null;
    value = '';

    const line = raw.trim();
    if (line === 'BEGIN:VEVENT') current = {};
    else if (line === 'END:VEVENT' && current) {
      if (current.dtstart && current.summary) events.push(current);
      current = null;
    } else if (current && line.includes(':')) {
      const i = line.indexOf(':');
      field = line.substring(0, i);
      value = line.substring(i + 1);
    }
  }
  return events;
}

function processField(event, field, value) {
  if (field.startsWith('DTSTART')) {
    const m = value.match(/(\d{8}T\d{6}Z?|\d{8})/);
    if (m) event.dtstart = parseICSDate(m[1]);
  } else if (field.startsWith('DTEND')) {
    const m = value.match(/(\d{8}T\d{6}Z?|\d{8})/);
    if (m) event.dtend = parseICSDate(m[1]);
  } else if (field === 'SUMMARY') event.summary = value.trim();
  else if (field === 'DESCRIPTION') event.description = value.trim();
  else if (field === 'UID') event.uid = value.trim();
}

function parseICSDate(s) {
  const date = `${s.substring(0, 4)}-${s.substring(4, 6)}-${s.substring(6, 8)}`;
  if (s.length === 8) return { date, isAllDay: true };
  const time = `${s.substring(9, 11)}:${s.substring(11, 13)}:${s.substring(13, 15)}`;
  return { date, time, isAllDay: false, timestamp: new Date(`${date}T${time}Z`).toISOString() };
}