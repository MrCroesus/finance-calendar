import fetch from 'node-fetch';
import fs from 'fs/promises';
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

async function fetchAndCacheCalendars() {
  console.log('🔄 Fetching economic calendars...\n');
  
  const calendars = [
    {
      name: 'BLS',
      url: 'https://www.bls.gov/schedule/news_release/bls.ics',
      cacheKey: 'bls_calendar'
    },
    {
      name: 'BEA',
      url: 'https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics',
      cacheKey: 'bea_calendar'
    }
  ];

  for (const cal of calendars) {
    try {
      console.log(`📥 Fetching ${cal.name} calendar from ${cal.url}...`);
      
      const response = await fetch(cal.url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; EarningsCalendar/1.0)',
          'Accept': 'text/calendar,*/*'
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const icsContent = await response.text();
      console.log(`  ✓ Downloaded ${icsContent.length} characters`);
      
      const events = parseICS(icsContent);
      console.log(`  ✓ Parsed ${events.length} events`);
      
      // Save to Supabase
      const { error } = await supabase
        .from('economic_calendar_cache')
        .upsert({
          cache_key: cal.cacheKey,
          events: events,
          last_updated: new Date().toISOString()
        }, {
          onConflict: 'cache_key'
        });

      if (error) throw error;
      console.log(`  ✓ Cached in Supabase`);
      
      // Save backup to file
      await fs.writeFile(
        `./data/${cal.cacheKey}.json`,
        JSON.stringify(events, null, 2)
      );
      console.log(`  ✓ Saved backup to data/${cal.cacheKey}.json`);
      console.log(`✅ ${cal.name} complete!\n`);
      
    } catch (error) {
      console.error(`❌ Failed to fetch ${cal.name}:`, error.message);
      
      // Try backup file
      try {
        const backup = await fs.readFile(`./data/${cal.cacheKey}.json`, 'utf-8');
        const events = JSON.parse(backup);
        
        await supabase
          .from('economic_calendar_cache')
          .upsert({
            cache_key: cal.cacheKey,
            events: events,
            last_updated: new Date().toISOString()
          });
        
        console.log(`  ⚠️  Using backup file (${events.length} events)\n`);
      } catch (backupError) {
        console.error(`  ❌ No backup available\n`);
      }
    }
  }
}

function parseICS(icsContent) {
  const events = [];
  const lines = icsContent.split(/\r?\n/);
  let currentEvent = null;
  let currentField = null;
  let currentValue = '';

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];

    // Handle line continuations (lines starting with space or tab)
    if (line.match(/^[ \t]/) && currentField) {
      currentValue += line.substring(1);
      continue;
    }

    // Process previous field if exists
    if (currentField && currentEvent) {
      processField(currentEvent, currentField, currentValue);
    }

    // Reset for new field
    currentField = null;
    currentValue = '';

    line = line.trim();

    if (line === 'BEGIN:VEVENT') {
      currentEvent = {};
    } else if (line === 'END:VEVENT' && currentEvent) {
      if (currentEvent.dtstart && currentEvent.summary) {
        events.push(currentEvent);
      }
      currentEvent = null;
    } else if (currentEvent && line.includes(':')) {
      const colonIndex = line.indexOf(':');
      currentField = line.substring(0, colonIndex);
      currentValue = line.substring(colonIndex + 1);
    }
  }

  return events;
}

function processField(event, field, value) {
  if (field.startsWith('DTSTART')) {
    const dateMatch = value.match(/(\d{8}T\d{6}Z?|\d{8})/);
    if (dateMatch) {
      event.dtstart = parseICSDate(dateMatch[1]);
    }
  } else if (field.startsWith('DTEND')) {
    const dateMatch = value.match(/(\d{8}T\d{6}Z?|\d{8})/);
    if (dateMatch) {
      event.dtend = parseICSDate(dateMatch[1]);
    }
  } else if (field === 'SUMMARY') {
    event.summary = value.trim();
  } else if (field === 'DESCRIPTION') {
    event.description = value.trim();
  } else if (field === 'UID') {
    event.uid = value.trim();
  }
}

function parseICSDate(dateStr) {
  if (dateStr.length === 8) {
    // Date only: YYYYMMDD
    return {
      date: `${dateStr.substring(0,4)}-${dateStr.substring(4,6)}-${dateStr.substring(6,8)}`,
      isAllDay: true
    };
  } else {
    // DateTime: YYYYMMDDTHHmmssZ
    const year = dateStr.substring(0, 4);
    const month = dateStr.substring(4, 6);
    const day = dateStr.substring(6, 8);
    const hour = dateStr.substring(9, 11);
    const minute = dateStr.substring(11, 13);
    const second = dateStr.substring(13, 15);
    
    return {
      date: `${year}-${month}-${day}`,
      time: `${hour}:${minute}:${second}`,
      isAllDay: false,
      timestamp: new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}Z`).toISOString()
    };
  }
}

// Run
fetchAndCacheCalendars()
  .then(() => {
    console.log('✨ All calendars updated successfully!');
    process.exit(0);
  })
  .catch(err => {
    console.error('💥 Fatal error:', err);
    process.exit(1);
  });