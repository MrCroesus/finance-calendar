try {
        console.log(`📊 Fetching ${ticker}...`);
        
        const data = await yahooFinance.quoteSummary(ticker, {
          modules: ['calendarEvents', 'price']
        });
        
        // ... rest stays the same
        
      } catch (error) {
        console.log(`  ❌ Error fetching ${ticker}:`);
        console.log(`     Error type: ${error.constructor.name}`);
        console.log(`     Error message: ${error.message}`);
        
        // If there's a response, show it
        if (error.result) {
          console.log(`     Yahoo response:`, JSON.stringify(error.result).substring(0, 200));
        }
        
        failed++;
        
        // Only wait if it's actually a rate limit
        if (error.message.includes('Too Many Requests') || 
            error.message.includes('429') ||
            error.message.includes('Rate limit')) {
          console.log('  ⏳ Rate limited, waiting 30 seconds...');
          await new Promise(resolve => setTimeout(resolve, 30000));
        } else {
          // For other errors, just wait 2 seconds and continue
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }