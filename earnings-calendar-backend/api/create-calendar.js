import { supabase } from '../lib/supabase.js';

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { tickers } = req.body;

    if (!tickers || !Array.isArray(tickers) || tickers.length === 0) {
      return res.status(400).json({ error: 'Invalid tickers array' });
    }

    if (tickers.length > 500) {
      return res.status(400).json({ error: 'Maximum 500 tickers allowed' });
    }

    const calendarId = Math.random().toString(36).substring(2, 15);

    const { data, error } = await supabase
      .from('calendars')
      .insert({
        id: calendarId,
        tickers: tickers.map(t => t.toUpperCase()),
        created_at: new Date().toISOString()
      })
      .select()
      .single();

    if (error) {
      console.error('Supabase error:', error);
      return res.status(500).json({ error: 'Failed to create calendar' });
    }

    res.status(200).json({
      calendarId,
      url: `/api/calendar/${calendarId}`
    });
  } catch (error) {
    console.error('Error creating calendar:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
}