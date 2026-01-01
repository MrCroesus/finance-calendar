// api/create-calendar.js
import { supabase } from '../lib/supabase.js';
import { randomBytes } from 'crypto';

export default async function handler(req, res) {
  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    return res.status(200).end();
  }

  // Set CORS headers for actual request
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { tickers, include_fomc, include_bls, include_bea } = req.body;

    // Validate input
    if (!Array.isArray(tickers)) {
      return res.status(400).json({ error: 'tickers must be an array' });
    }

    // Generate unique calendar ID
    const id = randomBytes(16).toString('hex');

    // Insert into database
    const { data, error } = await supabase
      .from('calendars')
      .insert({
        id: id,
        tickers: tickers,
        include_fomc: include_fomc || false,
        include_bls: include_bls || false,
        include_bea: include_bea || false,
        created_at: new Date().toISOString()
      })
      .select()
      .single();

    if (error) {
      console.error('Database error:', error);
      return res.status(500).json({ error: 'Failed to create calendar', details: error.message });
    }

    // Return calendar ID and subscription URL
    const baseUrl = process.env.VERCEL_URL 
      ? `https://${process.env.VERCEL_URL}`
      : req.headers.host 
        ? `https://${req.headers.host}`
        : 'http://localhost:3000';

    return res.status(201).json({
      id: data.id,
      subscriptionUrl: `${baseUrl}/api/calendar/${data.id}`,
      calendar: data
    });

  } catch (error) {
    console.error('Error creating calendar:', error);
    return res.status(500).json({ 
      error: 'Internal server error',
      message: error.message 
    });
  }
}