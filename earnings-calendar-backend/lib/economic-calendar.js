// FILE: lib/economic-calendar.js
// This module fetches economic calendar dates with API fallback to hardcoded data

// Hardcoded FOMC meeting dates for 2025-2026
const FOMC_MEETINGS_FALLBACK = [
  { date: '2025-01-29', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-03-19', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-05-07', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-06-18', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-07-30', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-09-17', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-11-05', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2025-12-17', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2026-01-28', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
  { date: '2026-03-18', title: 'FOMC Meeting', description: 'Federal Reserve interest rate decision' },
];

// Generate monthly BLS reports (Jobs Report - first Friday, CPI - mid-month)
function generateBLSReports() {
  const reports = [];
  const startYear = 2025;
  const endYear = 2026;
  
  for (let year = startYear; year <= endYear; year++) {
    for (let month = 0; month < 12; month++) {
      // Jobs Report - first Friday of the month
      const firstDay = new Date(year, month, 1);
      const firstFriday = new Date(year, month, 1 + (5 - firstDay.getDay() + 7) % 7);
      if (firstFriday.getDate() > 7) firstFriday.setDate(firstFriday.getDate() - 7);
      
      reports.push({
        date: firstFriday.toISOString().split('T')[0],
        title: 'Jobs Report',
        description: 'Bureau of Labor Statistics employment situation report (unemployment rate, jobs added)'
      });
      
      // CPI Report - typically around 13th-15th of the month
      const cpiDate = new Date(year, month, 13);
      reports.push({
        date: cpiDate.toISOString().split('T')[0],
        title: 'CPI Report',
        description: 'Bureau of Labor Statistics Consumer Price Index (inflation data)'
      });
    }
  }
  
  return reports;
}

// Generate quarterly BEA reports (GDP releases)
function generateBEAReports() {
  const reports = [];
  
  // GDP releases are typically last week of Jan, Apr, Jul, Oct (advance estimate)
  const gdpMonths = [
    { month: 0, day: 30, title: 'Q4 GDP (Advance)' },     // January - Q4 of previous year
    { month: 3, day: 30, title: 'Q1 GDP (Advance)' },     // April - Q1
    { month: 6, day: 31, title: 'Q2 GDP (Advance)' },     // July - Q2
    { month: 9, day: 30, title: 'Q3 GDP (Advance)' },     // October - Q3
  ];
  
  for (let year = 2025; year <= 2026; year++) {
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

// Try to fetch FOMC dates from an API (future enhancement)
async function fetchFOMCDates() {
  try {
    // Currently no free API available, use fallback
    // In the future, could scrape Federal Reserve website or use paid API
    return FOMC_MEETINGS_FALLBACK;
  } catch (error) {
    console.error('Error fetching FOMC dates, using fallback:', error);
    return FOMC_MEETINGS_FALLBACK;
  }
}

// Main export functions
export async function getFOMCDates() {
  return await fetchFOMCDates();
}

export function getBLSDates() {
  return generateBLSReports();
}

export function getBEADates() {
  return generateBEAReports();
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
  
  const uid = `economic-${event.title.replace(/\s+/g, '-')}-${dateStr}-${calendarId}@earningscalendar.com`;
  
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