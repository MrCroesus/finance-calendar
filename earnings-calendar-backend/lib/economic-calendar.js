// lib/economic-calendar.js
// Provides functions to fetch economic calendar events (FOMC, BLS, BEA)

import { supabase } from './supabase.js';

// ============================================================
// FOMC MEETINGS
// Update annually from: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
// ============================================================

const FOMC_MEETINGS = [
  // 2026
  { date: '2026-01-27', endDate: '2026-01-28', title: 'FOMC Meeting', hasSEP: false },
  { date: '2026-03-17', endDate: '2026-03-18', title: 'FOMC Meeting', hasSEP: true },
  { date: '2026-04-28', endDate: '2026-04-29', title: 'FOMC Meeting', hasSEP: false },
  { date: '2026-06-16', endDate: '2026-06-17', title: 'FOMC Meeting', hasSEP: true },
  { date: '2026-07-28', endDate: '2026-07-29', title: 'FOMC Meeting', hasSEP: false },
  { date: '2026-09-15', endDate: '2026-09-16', title: 'FOMC Meeting', hasSEP: true },
  { date: '2026-10-27', endDate: '2026-10-28', title: 'FOMC Meeting', hasSEP: false },
  { date: '2026-12-08', endDate: '2026-12-09', title: 'FOMC Meeting', hasSEP: true },
  
  // 2027 - ADD THESE DATES WHEN ANNOUNCED
  // Check which meetings have * on: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  // { date: '2026-01-27', endDate: '2026-01-28', title: 'FOMC Meeting', hasSEP: false },
  // { date: '2026-03-17', endDate: '2026-03-18', title: 'FOMC Meeting', hasSEP: true },
  // etc...
];

// ============================================================
// BLS EVENTS (Bureau of Labor Statistics)
// Fetched from: https://www.bls.gov/schedule/news_release/bls.ics
// Cached in Supabase by build script
// ============================================================

export async function getBLSEvents() {
  try {
    const { data, error } = await supabase
      .from('economic_calendar_cache')
      .select('events, last_updated')
      .eq('cache_key', 'bls_calendar')
      .single();

    if (error) {
      console.error('Error querying BLS cache:', error);
      return [];
    }

    if (data && data.events && Array.isArray(data.events)) {
      console.log(`✓ Loaded ${data.events.length} BLS events from cache (updated: ${data.last_updated})`);
      return data.events;
    }

    console.warn('⚠️  BLS events array is empty or invalid');
    return [];
    
  } catch (err) {
    console.error('Exception fetching BLS cache:', err);
    return [];
  }
}

// ============================================================
// BEA EVENTS (Bureau of Economic Analysis)
// Fetched from: https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics
// Cached in Supabase by build script
// ============================================================

export async function getBEAEvents() {
  try {
    const { data, error } = await supabase
      .from('economic_calendar_cache')
      .select('events, last_updated')
      .eq('cache_key', 'bea_calendar')
      .single();

    if (error) {
      console.error('Error querying BEA cache:', error);
      return [];
    }

    if (data && data.events && Array.isArray(data.events)) {
      console.log(`✓ Loaded ${data.events.length} BEA events from cache (updated: ${data.last_updated})`);
      return data.events;
    }

    console.warn('⚠️  BEA events array is empty or invalid');
    return [];
    
  } catch (err) {
    console.error('Exception fetching BEA cache:', err);
    return [];
  }
}

// ============================================================
// FOMC EVENTS
// Hardcoded - stable enough to not need caching
// ============================================================

export function getFOMCEvents() {
  const events = FOMC_MEETINGS.map(meeting => ({
    dtstart: { 
      date: meeting.startDate, 
      isAllDay: true 
    },
    dtend: { 
      date: meeting.endDate, 
      isAllDay: true 
    },
    summary: 'FOMC Meeting',
    description: meeting.hasSEP 
      ? 'Federal Reserve FOMC Meeting - Interest rate decision & Summary of Economic Projections (SEP)'
      : 'Federal Reserve FOMC Meeting - Interest rate decision',
    uid: `fomc-${meeting.startDate}@earnings-calendar`
  }));

  console.log(`✓ Loaded ${events.length} FOMC events`);
  return events;
}

// ============================================================
// HELPER: Convert event to iCalendar format
// ============================================================

export function eventToICS(event, calendarId) {
  const lines = [];
  
  lines.push('BEGIN:VEVENT');
  
  // UID - unique identifier
  const uid = event.uid || `${event.summary}-${event.dtstart.date}@${calendarId}`;
  lines.push(`UID:${uid}`);
  
  // DTSTART
  if (event.dtstart.isAllDay) {
    lines.push(`DTSTART;VALUE=DATE:${event.dtstart.date.replace(/-/g, '')}`);
  } else if (event.dtstart.timestamp) {
    // Use timestamp if available (already in UTC)
    const dt = new Date(event.dtstart.timestamp);
    lines.push(`DTSTART:${formatICSDateTime(dt)}`);
  } else if (event.dtstart.time) {
    // Use date + time
    const dt = new Date(`${event.dtstart.date}T${event.dtstart.time}Z`);
    lines.push(`DTSTART:${formatICSDateTime(dt)}`);
  } else {
    // Fallback to all-day
    lines.push(`DTSTART;VALUE=DATE:${event.dtstart.date.replace(/-/g, '')}`);
  }
  
  // DTEND (if exists)
  if (event.dtend) {
    if (event.dtend.isAllDay) {
      lines.push(`DTEND;VALUE=DATE:${event.dtend.date.replace(/-/g, '')}`);
    } else if (event.dtend.timestamp) {
      const dt = new Date(event.dtend.timestamp);
      lines.push(`DTEND:${formatICSDateTime(dt)}`);
    } else if (event.dtend.time) {
      const dt = new Date(`${event.dtend.date}T${event.dtend.time}Z`);
      lines.push(`DTEND:${formatICSDateTime(dt)}`);
    }
  }
  
  // SUMMARY
  lines.push(`SUMMARY:${escapeICSText(event.summary)}`);
  
  // DESCRIPTION (if exists)
  if (event.description) {
    lines.push(`DESCRIPTION:${escapeICSText(event.description)}`);
  }
  
  // DTSTAMP (current time)
  lines.push(`DTSTAMP:${formatICSDateTime(new Date())}`);
  
  lines.push('END:VEVENT');
  
  return lines.join('\r\n');
}

// Format date for iCalendar (YYYYMMDDTHHmmssZ)
function formatICSDateTime(date) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

// Escape special characters in iCalendar text
function escapeICSText(text) {
  if (!text) return '';
  return text
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/\n/g, '\\n');
}

// ============================================================
// HELPER: Get all economic events based on flags
// ============================================================

export async function getEconomicEvents(includeFOMC, includeBLS, includeBEA) {
  const allEvents = [];
  
  if (includeFOMC) {
    const fomcEvents = getFOMCEvents();
    allEvents.push(...fomcEvents);
  }
  
  if (includeBLS) {
    const blsEvents = await getBLSEvents();
    allEvents.push(...blsEvents);
  }
  
  if (includeBEA) {
    const beaEvents = await getBEAEvents();
    allEvents.push(...beaEvents);
  }
  
  // Sort by date
  allEvents.sort((a, b) => {
    const dateA = a.dtstart.timestamp || a.dtstart.date;
    const dateB = b.dtstart.timestamp || b.dtstart.date;
    return new Date(dateA) - new Date(dateB);
  });
  
  return allEvents;
}

// ============================================================
// HELPER: Filter events by date range
// ============================================================

export function filterEventsByDateRange(events, startDate, endDate) {
  return events.filter(event => {
    const eventDate = new Date(event.dtstart.timestamp || event.dtstart.date);
    const start = startDate ? new Date(startDate) : new Date(0);
    const end = endDate ? new Date(endDate) : new Date('2099-12-31');
    
    return eventDate >= start && eventDate <= end;
  });
}

// ============================================================
// EXPORTS
// ============================================================

export default {
  getBLSEvents,
  getBEAEvents,
  getFOMCEvents,
  getEconomicEvents,
  eventToICS,
  filterEventsByDateRange
};