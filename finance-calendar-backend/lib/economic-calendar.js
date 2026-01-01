// lib/economic-calendar.js
import { supabase } from './supabase.js';

// FOMC meetings - update annually from federalreserve.gov
const FOMC_MEETINGS = [
  // 2026
  { startDate: '2026-01-27', endDate: '2026-01-28', title: 'FOMC Meeting', hasSEP: false },
  { startDate: '2026-03-17', endDate: '2026-03-18', title: 'FOMC Meeting', hasSEP: true },
  { startDate: '2026-04-28', endDate: '2026-04-29', title: 'FOMC Meeting', hasSEP: false },
  { startDate: '2026-06-16', endDate: '2026-06-17', title: 'FOMC Meeting', hasSEP: true },
  { startDate: '2026-07-28', endDate: '2026-07-29', title: 'FOMC Meeting', hasSEP: false },
  { startDate: '2026-09-15', endDate: '2026-09-16', title: 'FOMC Meeting', hasSEP: true },
  { startDate: '2026-10-27', endDate: '2026-10-28', title: 'FOMC Meeting', hasSEP: false },
  { startDate: '2026-12-08', endDate: '2026-12-09', title: 'FOMC Meeting', hasSEP: true },
  
  // 2027 - ADD THESE DATES WHEN ANNOUNCED
  // Check which meetings have * on: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  // { startDate: '2026-01-27', endDate: '2026-01-28', title: 'FOMC Meeting', hasSEP: false },
  // { startDate: '2026-03-17', endDate: '2026-03-18', title: 'FOMC Meeting', hasSEP: true },
  // etc...
];

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
      console.log(`✓ Loaded ${data.events.length} BLS events from cache`);
      return data.events;
    }

    console.warn('⚠️  BLS events array is empty');
    return [];
    
  } catch (err) {
    console.error('Exception fetching BLS cache:', err);
    return [];
  }
}

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
      console.log(`✓ Loaded ${data.events.length} BEA events from cache`);
      return data.events;
    }

    console.warn('⚠️  BEA events array is empty');
    return [];
    
  } catch (err) {
    console.error('Exception fetching BEA cache:', err);
    return [];
  }
}

export function getFOMCEvents() {
  const events = FOMC_MEETINGS.map(meeting => {
    // For multi-day all-day events, iCalendar DTEND is exclusive
    // So we need to add 1 day to the end date
    const endDate = new Date(meeting.endDate);
    endDate.setDate(endDate.getDate() + 1);
    const endDateStr = endDate.toISOString().split('T')[0];

    return {
      dtstart: { 
        date: meeting.startDate, 
        isAllDay: true 
      },
      dtend: { 
        date: endDateStr, 
        isAllDay: true 
      },
      summary: 'FOMC Meeting',
      description: meeting.hasSEP 
        ? 'Federal Reserve FOMC Meeting - Interest rate decision & Summary of Economic Projections (SEP)'
        : 'Federal Reserve FOMC Meeting - Interest rate decision',
      uid: `fomc-${meeting.startDate}@earnings-calendar`
    };
  });

  console.log(`✓ Loaded ${events.length} FOMC events`);
  return events;
}

export function eventToICS(event, calendarId) {
  const lines = [];
  
  lines.push('BEGIN:VEVENT');
  
  // Validate event has required fields
  if (!event || !event.dtstart || !event.summary) {
    console.error('Invalid event:', event);
    return ''; // Skip invalid events
  }
  
  const uid = event.uid || `${event.summary}-${event.dtstart.date}@${calendarId}`;
  lines.push(`UID:${uid}`);
  
  // DTSTART
  if (event.dtstart.isAllDay && event.dtstart.date) {
    lines.push(`DTSTART;VALUE=DATE:${event.dtstart.date.replace(/-/g, '')}`);
  } else if (event.dtstart.timestamp) {
    const dt = new Date(event.dtstart.timestamp);
    lines.push(`DTSTART:${formatICSDateTime(dt)}`);
  } else if (event.dtstart.time && event.dtstart.date) {
    const dt = new Date(`${event.dtstart.date}T${event.dtstart.time}Z`);
    lines.push(`DTSTART:${formatICSDateTime(dt)}`);
  } else if (event.dtstart.date) {
    lines.push(`DTSTART;VALUE=DATE:${event.dtstart.date.replace(/-/g, '')}`);
  } else {
    console.error('Event missing date:', event);
    return ''; // Skip events without a valid date
  }
  
  // DTEND
  if (event.dtend) {
    if (event.dtend.isAllDay && event.dtend.date) {
      lines.push(`DTEND;VALUE=DATE:${event.dtend.date.replace(/-/g, '')}`);
    } else if (event.dtend.timestamp) {
      const dt = new Date(event.dtend.timestamp);
      lines.push(`DTEND:${formatICSDateTime(dt)}`);
    } else if (event.dtend.time && event.dtend.date) {
      const dt = new Date(`${event.dtend.date}T${event.dtend.time}Z`);
      lines.push(`DTEND:${formatICSDateTime(dt)}`);
    }
  }
  
  lines.push(`SUMMARY:${escapeICSText(event.summary)}`);
  
  if (event.description) {
    lines.push(`DESCRIPTION:${escapeICSText(event.description)}`);
  }
  
  lines.push(`DTSTAMP:${formatICSDateTime(new Date())}`);
  lines.push('END:VEVENT');
  
  return lines.join('\r\n');
}

function formatICSDateTime(date) {
  return date.toISOString().replace(/[-:]/g, '').replace(/\.\d{3}/, '');
}

function escapeICSText(text) {
  if (!text) return '';
  // Only escape semicolons and newlines for iCalendar format
  // Commas and backslashes don't need escaping in most calendar apps
  return text
    .replace(/\n/g, '\\n')
    .replace(/;/g, '\\;');
}

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
  
  allEvents.sort((a, b) => {
    const dateA = a.dtstart.timestamp || a.dtstart.date;
    const dateB = b.dtstart.timestamp || b.dtstart.date;
    return new Date(dateA) - new Date(dateB);
  });
  
  return allEvents;
}

export function filterEventsByDateRange(events, startDate, endDate) {
  return events.filter(event => {
    const eventDate = new Date(event.dtstart.timestamp || event.dtstart.date);
    const start = startDate ? new Date(startDate) : new Date(0);
    const end = endDate ? new Date(endDate) : new Date('2099-12-31');
    
    return eventDate >= start && eventDate <= end;
  });
}

export default {
  getBLSEvents,
  getBEAEvents,
  getFOMCEvents,
  getEconomicEvents,
  eventToICS,
  filterEventsByDateRange
};