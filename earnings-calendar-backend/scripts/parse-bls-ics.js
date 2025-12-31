// FILE: scripts/parse-bls-ics.js
// Run this script to convert BLS ICS file to JSON format
// Usage: node scripts/parse-bls-ics.js path/to/bls.ics

import fs from 'fs';
import path from 'path';

function parseICS(icsContent) {
  const events = [];
  const lines = icsContent.split('\n').map(line => line.trim());
  
  let currentEvent = {};
  let inEvent = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (line === 'BEGIN:VEVENT') {
      inEvent = true;
      currentEvent = {};
    } else if (line === 'END:VEVENT') {
      inEvent = false;
      
      // Determine description based on title
      if (currentEvent.title && currentEvent.date) {
        let description = 'Bureau of Labor Statistics report';
        
        if (currentEvent.title.includes('Employment Situation')) {
          description = 'Jobs report (unemployment rate, nonfarm payrolls)';
        } else if (currentEvent.title.includes('Consumer Price Index')) {
          description = 'CPI inflation data';
        } else if (currentEvent.title.includes('Producer Price Index')) {
          description = 'PPI wholesale inflation data';
        } else if (currentEvent.title.includes('Real Earnings')) {
          description = 'Real earnings data';
        } else if (currentEvent.title.includes('Job Openings')) {
          description = 'JOLTS job openings and labor turnover';
        } else if (currentEvent.title.includes('Productivity')) {
          description = 'Labor productivity and costs';
        } else if (currentEvent.title.includes('Import')) {
          description = 'Import and export price indexes';
        } else if (currentEvent.title.includes('Employment Cost Index')) {
          description = 'Employment cost index';
        } else if (currentEvent.title.includes('Metropolitan Area Employment')) {
          description = 'Metropolitan area employment and unemployment';
        }
        
        currentEvent.description = description;
        events.push({ ...currentEvent });
      }
    } else if (inEvent) {
      // Parse DTSTART - handle multiple formats and extract time
      if (line.startsWith('DTSTART')) {
        // Extract date and time from various formats:
        // DTSTART;TZID=US-Eastern:20250103T100000
        const match = line.match(/DTSTART[^:]*:(\d{8})T?(\d{6})?/);
        if (match) {
          const dateStr = match[1];
          currentEvent.date = `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
          
          // Extract time if present (format: HHMMSS)
          if (match[2]) {
            const timeStr = match[2];
            const hours = timeStr.slice(0,2);
            const minutes = timeStr.slice(2,4);
            currentEvent.time = `${hours}:${minutes}`;
          }
        }
      }
      
      // Parse SUMMARY
      if (line.startsWith('SUMMARY:')) {
        currentEvent.title = line.substring('SUMMARY:'.length).trim();
      }
    }
  }
  
  // Sort by date
  events.sort((a, b) => a.date.localeCompare(b.date));
  
  return events;
}

function generateJavaScriptCode(events) {
  let code = '// BLS SCHEDULE - Auto-generated from ICS file\n';
  code += '// Last updated: ' + new Date().toISOString().split('T')[0] + '\n';
  code += 'const BLS_SCHEDULE = [\n';
  
  let currentYear = null;
  let currentMonth = null;
  
  events.forEach(event => {
    const eventDate = new Date(event.date);
    const year = eventDate.getFullYear();
    const month = eventDate.getMonth();
    
    // Add year comment
    if (year !== currentYear) {
      if (currentYear !== null) code += '\n';
      code += `  // ${year}\n`;
      currentYear = year;
      currentMonth = null;
    }
    
    // Add month comment
    if (month !== currentMonth) {
      const monthName = eventDate.toLocaleDateString('en-US', { month: 'long' });
      code += `  // ${monthName}\n`;
      currentMonth = month;
    }
    
    // Include time if available
    const timeStr = event.time ? `, time: '${event.time}'` : '';
    code += `  { date: '${event.date}'${timeStr}, title: '${event.title}', description: '${event.description}' },\n`;
  });
  
  code += '];\n';
  
  return code;
}

// Main execution
const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('Usage: node scripts/parse-bls-ics.js <path-to-bls.ics>');
  console.error('');
  console.error('Example:');
  console.error('  node scripts/parse-bls-ics.js ~/Downloads/bls.ics');
  console.error('');
  console.error('Download BLS schedule from:');
  console.error('  https://www.bls.gov/schedule/news_release/bls.ics');
  process.exit(1);
}

const icsFilePath = args[0];

try {
  console.log(`Reading ICS file: ${icsFilePath}`);
  const icsContent = fs.readFileSync(icsFilePath, 'utf-8');
  
  console.log('Parsing ICS content...');
  const events = parseICS(icsContent);
  
  console.log(`Found ${events.length} events`);
  
  // Generate JavaScript code
  const jsCode = generateJavaScriptCode(events);
  
  // Output to console
  console.log('\n=== Copy this into lib/economic-calendar.js ===\n');
  console.log(jsCode);
  
  // Optionally save to file
  const outputPath = 'scripts/bls-schedule-output.js';
  fs.writeFileSync(outputPath, jsCode);
  console.log(`\n✅ Also saved to: ${outputPath}`);
  console.log('\nNext steps:');
  console.log('1. Copy the BLS_SCHEDULE array above');
  console.log('2. Replace the BLS_SCHEDULE array in lib/economic-calendar.js');
  console.log('3. Deploy your changes');
  
} catch (error) {
  console.error('Error:', error.message);
  process.exit(1);
}

// =============================================================================
// INSTRUCTIONS FOR USE
// =============================================================================
/*

## How to Update BLS Schedule:

### Step 1: Download the ICS file
Go to: https://www.bls.gov/schedule/news_release/bls.ics
Save it to your computer (e.g., ~/Downloads/bls.ics)

### Step 2: Run this script
```bash
node scripts/parse-bls-ics.js ~/Downloads/bls.ics
```

### Step 3: Copy the output
The script will print the formatted JavaScript code.
Copy the entire BLS_SCHEDULE array.

### Step 4: Update your code
Open lib/economic-calendar.js
Replace the existing BLS_SCHEDULE array with the new one.

### Step 5: Deploy
```bash
git add lib/economic-calendar.js
git commit -m "Update BLS schedule for 2026"
git push origin main
```

Done! Your calendar now has accurate BLS dates.

*/