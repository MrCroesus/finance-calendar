export default async function handler(req, res) {
  console.log('🧪 Test log message');
  console.log('Time:', new Date().toISOString());
  console.log('Headers:', req.headers);
  
  return res.status(200).json({ 
    message: 'Test successful',
    timestamp: new Date().toISOString()
  });
}