// FILE: lib/economic-calendar.js
// Economic calendar data with multiple sources

import fetch from 'node-fetch';
import ICAL from 'ical.js';

// ============================================================================
// FOMC MEETINGS - Hardcoded (update annually)
// ============================================================================
// Get dates from: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
// * = Meeting associated with Summary of Economic Projections
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

// ============================================================================
// BLS REPORTS - Fetch from official BLS calendar
// ============================================================================
async function fetchBLSDates() {
  try {
    console.log('Fetching BLS calendar from official source...');
    const response = await fetch('https://www.bls.gov/schedule/news_release/bls.ics');
    
    if (!response.ok) {
      throw new Error(`BLS calendar fetch failed: ${response.status}`);
    }
    
    const icsText = await response.text();
    
    // Parse ICS file using ical.js
    const jcalData = ICAL.parse(icsText);
    const comp = new ICAL.Component(jcalData);
    const vevents = comp.getAllSubcomponents('vevent');
    
    const blsEvents = [];
    const now = new Date();
    const twoYearsFromNow = new Date(now.getFullYear() + 2, now.getMonth(), now.getDate());
    
    for (const vevent of vevents) {
      const event = new ICAL.Event(vevent);
      const startDate = event.startDate.toJSDate();
      
      // Only include future events (within next 2 years)
      if (startDate >= now && startDate <= twoYearsFromNow) {
        blsEvents.push({
          date: startDate.toISOString().split('T')[0],
          title: event.summary || 'BLS Report',
          description: (event.description || 'Bureau of Labor Statistics economic report')
            .replace(/\n/g, ' ')
            .substring(0, 200) // Clean up description
        });
      }
    }
    
    console.log(`Fetched ${blsEvents.length} BLS events from official calendar`);
    return blsEvents;
    
  } catch (error) {
    console.error('Error fetching BLS calendar:', error);
    console.log('Using fallback: generating basic BLS schedule');
    
    // Fallback: Generate basic monthly schedule
    return generateBLSFallback();
  }
}

function generateBLSFallback() {
  const reports = [];
  const startYear = new Date().getFullYear();
  const endYear = startYear + 1;
  
  for (let year = startYear; year <= endYear; year++) {
    for (let month = 0; month < 12; month++) {
      // Jobs Report - first Friday of the month
      const firstDay = new Date(year, month, 1);
      let firstFriday = new Date(year, month, 1);
      while (firstFriday.getDay() !== 5) {
        firstFriday.setDate(firstFriday.getDate() + 1);
      }
      
      reports.push({
        date: firstFriday.toISOString().split('T')[0],
        title: 'Employment Situation Report',
        description: 'Bureau of Labor Statistics employment report (unemployment rate, jobs added)'
      });
    }
  }
  
  return reports;
}

// ============================================================================
// BEA REPORTS - Scrape from FRED GDP calendar
// ============================================================================
async function fetchBEADates() {
  try {
    console.log('Fetching BEA/GDP dates from FRED...');
    
    const currentYear = new Date().getFullYear();
    const startYear = currentYear;
    const endYear = currentYear + 1;
    
    const allGDPEvents = [];
    
    // Fetch data for current and next year
    for (let year = startYear; year <= endYear; year++) {
      const url = `https://fred.stlouisfed.org/releases/calendar?od=asc&rid=53&ve=${year}-12-31&view=year&vs=${year}-01-01`;
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`FRED fetch failed: ${response.status}`);
      }
      
      const html = await response.text();
      
      // Parse HTML to extract dates
      // Look for patterns like: <td class="date">2025-01-30</td>
      const dateRegex = /<td class="date">(\d{4}-\d{2}-\d{2})<\/td>/g;
      const titleRegex = /<td class="release-name">(.*?)<\/td>/gs;
      
      const dates = [...html.matchAll(dateRegex)].map(m => m[1]);
      const titles = [...html.matchAll(titleRegex)].map(m => 
        m[1].replace(/<[^>]*>/g, '').trim()
      );
      
      // Combine dates and titles
      for (let i = 0; i < Math.min(dates.length, titles.length); i++) {
        allGDPEvents.push({
          date: dates[i],
          title: titles[i] || 'GDP Report',
          description: 'Bureau of Economic Analysis Gross Domestic Product report'
        });
      }
    }
    
    console.log(`Fetched ${allGDPEvents.length} GDP release dates from FRED`);
    return allGDPEvents;
    
  } catch (error) {
    console.error('Error fetching BEA/GDP dates from FRED:', error);
    console.log('Using fallback: generating quarterly GDP schedule');
    
    return generateBEAFallback();
  }
}

function generateBEAFallback() {
  const reports = [];
  const currentYear = new Date().getFullYear();
  
  // GDP releases are typically last week of Jan, Apr, Jul, Oct
  const gdpMonths = [
    { month: 0, day: 30, title: 'GDP - Advance Estimate' },
    { month: 3, day: 30, title: 'GDP - Advance Estimate' },
    { month: 6, day: 31, title: 'GDP - Advance Estimate' },
    { month: 9, day: 30, title: 'GDP - Advance Estimate' },
  ];
  
  for (let year = currentYear; year <= currentYear + 1; year++) {
    gdpMonths.forEach(({ month, day, title }) => {
      reports.push({
        date: new Date(year, month, day).toISOString().split('T')[0],
        title: title,
        description: 'Bureau of Economic Analysis Gross Domestic Product report'
      });
    });
  }
  
  return reports;
}

// ============================================================================
// Main export functions
// ============================================================================
export async function getFOMCDates() {
  // Generate descriptions dynamically based on hasSEP flag
  return FOMC_MEETINGS.map(meeting => ({
    ...meeting,
    description: meeting.hasSEP 
      ? 'Interest rate decision & Summary of Economic Projections'
      : 'Interest rate decision'
  }));
}

export async function getBLSDates() {
  return await fetchBLSDates();
}

export async function getBEADates() {
  return await fetchBEADates();
}

// Helper function to format dates for iCalendar
export function formatEconomicEvent(event, calendarId) {
  const date = new Date(event.date);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const dateStr = `${year}${month}${day}`;
  
  const now = new Date();
  const timestamp = dateStr + 'T' + 
    now.toISOString().split('T')[1].replace(/[-:]/g, '').split('.')[0] + 'Z';
  
  // Create a clean UID
  const cleanTitle = event.title.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
  const uid = `economic-${cleanTitle}-${dateStr}-${calendarId}@financecalendar.com`;
  
  return `BEGIN:VEVENT
UID:${uid}
DTSTAMP:${timestamp}
DTSTART;VALUE=DATE:${dateStr}
SUMMARY:${event.title}
DESCRIPTION:${event.description}
STATUS:CONFIRMED
TRANSP:TRANSPARENT
END:VEVENT`;
}