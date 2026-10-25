// api/cron/refresh.js
import { refreshStocks, refreshEconomicCalendars, refreshEarnings } from '../../lib/refresh.js';

export default async function handler(req, res) {
  // Vercel sends "Authorization: Bearer <CRON_SECRET>" on cron invocations
  if (!process.env.CRON_SECRET || req.headers.authorization !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const start = Date.now();
  const deadline = start + 240 * 1000; // stop starting new tickers after 4 minutes
  const result = {};

  try {
    result.stocks = await refreshStocks();
  } catch (err) {
    result.stocks = { error: err.message };
  }

  result.economic = await refreshEconomicCalendars();

  try {
    result.earnings = await refreshEarnings(deadline);
  } catch (err) {
    result.earnings = { error: err.message };
  }

  result.seconds = Math.round((Date.now() - start) / 1000);
  console.log('Cron refresh result:', JSON.stringify(result));
  return res.status(200).json(result);
}