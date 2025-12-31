// FILE: lib/economic-calendar.js
// Economic calendar data with multiple sources

import fetch from 'node-fetch';

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
// BLS REPORTS - Load from downloaded schedule file
// ============================================================================
// Download schedule annually from: https://www.bls.gov/schedule/news_release/bls.ics
// Save as: lib/bls-schedule.json
// 
// To convert ICS to JSON:
// 1. Download https://www.bls.gov/schedule/news_release/bls.ics
// 2. Open in a text editor
// 3. Extract dates and titles manually, or use online ICS parser
// 4. Format as JSON array (see structure below)

const BLS_SCHEDULE = [
  // 2025 - UPDATE THESE DATES ANNUALLY from BLS website
  // Download from: https://www.bls.gov/schedule/news_release/bls.ics
  // 2025
  // January
  { date: '2025-01-03', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-01-07', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-01-10', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-01-14', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-01-15', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-01-15', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-01-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-01-17', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-01-22', time: '10:00', title: 'Usual Weekly Earnings of Wage and Salary Workers', description: 'Bureau of Labor Statistics report' },
  { date: '2025-01-28', time: '10:00', title: 'Union Membership (Annual)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-01-28', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-01-29', time: '10:00', title: 'Quarterly Data Series on Business Employment Dynamics', description: 'Bureau of Labor Statistics report' },
  { date: '2025-01-31', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  // February
  { date: '2025-02-04', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-02-05', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-02-06', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-02-07', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-02-12', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-02-12', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-02-13', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-02-14', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-02-19', time: '10:00', title: 'County Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2025-02-19', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-02-20', time: '10:00', title: 'Major Work Stoppages (Annual)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-02-25', time: '10:00', title: 'Persons with a Disability: Labor Force Characteristics', description: 'Bureau of Labor Statistics report' },
  // March
  { date: '2025-03-05', time: '10:00', title: 'County Employment and Wages Full Data Update', description: 'Bureau of Labor Statistics report' },
  { date: '2025-03-05', time: '10:00', title: 'State Unemployment (Annual)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-03-06', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-03-07', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-03-11', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-03-12', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-03-12', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-03-13', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-03-14', time: '10:00', title: 'Employer Costs for Employee Compensation', description: 'Bureau of Labor Statistics report' },
  { date: '2025-03-17', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-03-18', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-03-20', time: '10:00', title: 'Employment Situation of Veterans', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-03-20', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-03-21', time: '10:00', title: 'Total Factor Productivity', description: 'Labor productivity and costs' },
  { date: '2025-03-21', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-03-28', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-01', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  // April
  { date: '2025-04-02', time: '10:00', title: 'Occupational Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-04', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-04-09', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-04-10', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-04-10', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-04-11', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-04-15', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-04-16', time: '10:00', title: 'Usual Weekly Earnings of Wage and Salary Workers', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-16', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-04-18', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-22', time: '10:00', title: 'College Enrollment and Work Activity of High School Graduates', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-23', time: '10:00', title: 'Employment Characteristics of Families', description: 'Bureau of Labor Statistics report' },
  { date: '2025-04-24', time: '10:00', title: 'Productivity and Costs by Industry: Manufacturing and Mining Industries', description: 'Labor productivity and costs' },
  { date: '2025-04-29', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-04-29', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-04-30', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  // May
  { date: '2025-05-02', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-05-07', time: '10:00', title: 'Quarterly Data Series on Business Employment Dynamics', description: 'Bureau of Labor Statistics report' },
  { date: '2025-05-08', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-05-13', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-05-13', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-05-15', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-05-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-05-20', time: '10:00', title: 'Labor Force Characteristics of Foreign-born Workers', description: 'Bureau of Labor Statistics report' },
  { date: '2025-05-20', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-05-21', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-05-28', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-05-29', time: '10:00', title: 'Productivity and Costs by Industry: Wholesale Trade and Retail Trade', description: 'Labor productivity and costs' },
  { date: '2025-05-29', time: '10:00', title: 'Productivity by State', description: 'Labor productivity and costs' },
  // June
  { date: '2025-06-03', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-06-04', time: '10:00', title: 'County Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2025-06-05', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-06-06', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-06-11', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-06-11', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-06-12', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-06-13', time: '10:00', title: 'Employer Costs for Employee Compensation', description: 'Bureau of Labor Statistics report' },
  { date: '2025-06-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-06-18', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-06-24', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-06-26', time: '10:00', title: 'Productivity and Costs by Industry: Selected Service-Providing Industries', description: 'Labor productivity and costs' },
  { date: '2025-06-26', time: '10:00', title: 'American Time Use Survey', description: 'Bureau of Labor Statistics report' },
  { date: '2025-07-01', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  // July
  { date: '2025-07-02', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-07-03', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-07-15', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-07-15', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-07-16', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-07-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-07-18', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-07-22', time: '10:00', title: 'Usual Weekly Earnings of Wage and Salary Workers', description: 'Bureau of Labor Statistics report' },
  { date: '2025-07-23', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-07-29', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-07-30', time: '10:00', title: 'Quarterly Data Series on Business Employment Dynamics', description: 'Bureau of Labor Statistics report' },
  { date: '2025-07-30', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-07-31', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  { date: '2025-08-01', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  // August
  { date: '2025-08-07', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-08-12', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-08-12', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-08-13', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-08-14', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-08-15', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-08-19', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-08-21', time: '10:00', title: 'Summer Youth Labor Force', description: 'Bureau of Labor Statistics report' },
  { date: '2025-08-26', time: '10:00', title: 'Number of Jobs, Labor Market Experience, Marital Status, and Health for those Born 1957-1964', description: 'Bureau of Labor Statistics report' },
  { date: '2025-08-27', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-08-28', time: '10:00', title: 'Total Factor Productivity for Detailed Industries', description: 'Labor productivity and costs' },
  { date: '2025-08-28', time: '10:00', title: 'Employment Projections and Occupational Outlook Handbook', description: 'Bureau of Labor Statistics report' },
  // September
  { date: '2025-09-03', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-09-04', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2025-09-05', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-09-09', time: '10:00', title: 'Current Employment Statistics Preliminary Benchmark (National)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-09', time: '10:00', title: 'County Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-09', time: '10:00', title: 'Current Employment Statistics Preliminary Benchmark (State and Area)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-10', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2025-09-11', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-09-11', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-09-12', time: '10:00', title: 'Employer Costs for Employee Compensation', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-09-17', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-09-19', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-25', time: '10:00', title: 'Unpaid Eldercare in the United States', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-25', time: '10:00', title: 'Employee Benefits in the United States', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-30', time: '10:00', title: 'People with Health Conditions or Difficulties that Limit Work', description: 'Bureau of Labor Statistics report' },
  { date: '2025-09-30', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-10-01', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  // October
  { date: '2025-10-24', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  // November
  { date: '2025-11-20', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-11-21', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-11-25', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  // December
  { date: '2025-12-02', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-12-03', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2025-12-04', time: '10:00', title: 'Usual Weekly Earnings of Wage and Salary Workers', description: 'Bureau of Labor Statistics report' },
  { date: '2025-12-09', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2025-12-10', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  { date: '2025-12-11', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2025-12-16', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2025-12-17', time: '10:00', title: 'Quarterly Data Series on Business Employment Dynamics', description: 'Bureau of Labor Statistics report' },
  { date: '2025-12-17', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2025-12-18', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2025-12-18', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2025-12-19', time: '10:00', title: 'County Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2025-12-19', time: '10:00', title: 'Total Factor Productivity for Major Industries', description: 'Labor productivity and costs' },
  { date: '2025-12-19', time: '10:00', title: 'Consumer Expenditures', description: 'Bureau of Labor Statistics report' },
  { date: '2025-12-30', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },

  // 2026
  // January
  { date: '2026-01-07', time: '10:00', title: 'Job Openings and Labor Turnover Survey', description: 'JOLTS job openings and labor turnover' },
  { date: '2026-01-07', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2026-01-08', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-01-09', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-01-13', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-01-13', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-01-14', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-01-15', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-01-16', time: '10:00', title: 'Occupational Requirements in the United States', description: 'Bureau of Labor Statistics report' },
  { date: '2026-01-16', time: '10:00', title: 'Metropolitan Area Employment and Unemployment (Monthly)', description: 'Metropolitan area employment and unemployment' },
  { date: '2026-01-22', time: '10:00', title: 'Employer-Reported Workplace Injuries and Illnesses (Annual)', description: 'Bureau of Labor Statistics report' },
  { date: '2026-01-22', time: '10:00', title: 'Work Experience of the Population (Annual)', description: 'Bureau of Labor Statistics report' },
  { date: '2026-01-27', time: '10:00', title: 'State Job Openings and Labor Turnover', description: 'JOLTS job openings and labor turnover' },
  { date: '2026-01-27', time: '10:00', title: 'State Employment and Unemployment (Monthly)', description: 'Bureau of Labor Statistics report' },
  { date: '2026-01-29', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-01-30', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  // February
  { date: '2026-02-05', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-02-06', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-02-10', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-02-10', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  { date: '2026-02-11', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-02-11', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-02-12', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-02-18', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-02-19', time: '10:00', title: 'Census of Fatal Occupational Injuries', description: 'Bureau of Labor Statistics report' },
  { date: '2026-02-26', time: '10:00', title: 'Quarterly Data Series on Business Employment Dynamics', description: 'Bureau of Labor Statistics report' },
  // March
  { date: '2026-03-05', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-03-06', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-03-10', time: '10:00', title: 'County Employment and Wages', description: 'Bureau of Labor Statistics report' },
  { date: '2026-03-11', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-03-11', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-03-12', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-03-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // April
  { date: '2026-04-03', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-04-10', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-04-10', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-04-14', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-04-15', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-04-30', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  // May
  { date: '2026-05-07', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-05-08', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-05-12', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-05-12', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-05-13', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-05-14', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // June
  { date: '2026-06-04', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-06-05', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-06-10', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-06-10', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-06-11', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-06-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // July
  { date: '2026-07-02', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-07-14', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-07-14', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-07-15', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-07-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-07-31', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  // August
  { date: '2026-08-06', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-08-07', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-08-12', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-08-12', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-08-13', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-08-18', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // September
  { date: '2026-09-03', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-09-04', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-09-10', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-09-11', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-09-11', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-09-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // October
  { date: '2026-10-02', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-10-14', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-10-14', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-10-15', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-10-16', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  { date: '2026-10-30', time: '08:30', title: 'Employment Cost Index', description: 'Employment cost index' },
  // November
  { date: '2026-11-05', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-11-06', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-11-10', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-11-10', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-11-13', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-11-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },
  // December
  { date: '2026-12-04', time: '08:30', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  { date: '2026-12-08', time: '08:30', title: 'Productivity and Costs', description: 'Labor productivity and costs' },
  { date: '2026-12-10', time: '08:30', title: 'Consumer Price Index', description: 'CPI inflation data' },
  { date: '2026-12-10', time: '08:30', title: 'Real Earnings', description: 'Real earnings data' },
  { date: '2026-12-15', time: '08:30', title: 'Producer Price Index', description: 'PPI wholesale inflation data' },
  { date: '2026-12-17', time: '08:30', title: 'U.S. Import and Export Price Indexes', description: 'Import and export price indexes' },  
  
  // 2026 - ADD DATES HERE WHEN PUBLISHED
  // Download from: https://www.bls.gov/schedule/news_release/bls.ics
  // { date: '2026-01-09', title: 'Employment Situation', description: 'Jobs report (unemployment rate, nonfarm payrolls)' },
  // etc...
];

async function fetchBLSDates() {
  console.log(`Loaded ${BLS_SCHEDULE.length} BLS events from hardcoded schedule`);
  
  // Filter to only show future dates
  const now = new Date();
  const futureEvents = BLS_SCHEDULE.filter(event => {
    const eventDate = new Date(event.date);
    return eventDate >= now;
  });
  
  return futureEvents;
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
      
      // Parse HTML structure:
      // <span style="font-weight: bold;">Thursday January 30, 2025</span>
      // followed by
      // <a href="/release?rid=53">Gross Domestic Product</a>
      
      // Match date headers like "Thursday January 30, 2025"
      const dateHeaderRegex = /<span style="font-weight: bold;">(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+(\w+)\s+(\d{1,2}),\s+(\d{4})<\/span>/g;
      const dateMatches = [...html.matchAll(dateHeaderRegex)];
      
      for (const match of dateMatches) {
        const monthName = match[1]; // e.g., "January"
        const day = match[2]; // e.g., "30"
        const year = match[3]; // e.g., "2025"
        
        // Convert month name to number
        const monthMap = {
          'January': '01', 'February': '02', 'March': '03', 'April': '04',
          'May': '05', 'June': '06', 'July': '07', 'August': '08',
          'September': '09', 'October': '10', 'November': '11', 'December': '12'
        };
        
        const month = monthMap[monthName];
        if (month) {
          const dateStr = `${year}-${month}-${day.padStart(2, '0')}`;
          
          allGDPEvents.push({
            date: dateStr,
            time: '08:30', // GDP releases are at 8:30 AM ET
            title: 'Gross Domestic Product',
            description: 'Bureau of Economic Analysis GDP report'
          });
        }
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
    { month: 0, day: 30, title: 'Gross Domestic Product' },
    { month: 3, day: 30, title: 'Gross Domestic Product' },
    { month: 6, day: 31, title: 'Gross Domestic Product' },
    { month: 9, day: 30, title: 'Gross Domestic Product' },
  ];
  
  for (let year = currentYear; year <= currentYear + 1; year++) {
    gdpMonths.forEach(({ month, day, title }) => {
      reports.push({
        date: new Date(year, month, day).toISOString().split('T')[0],
        time: '08:30',
        title: title,
        description: 'Bureau of Economic Analysis GDP report'
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
  
  // Handle multi-day events (FOMC meetings)
  let dtend = '';
  if (event.endDate) {
    const endDate = new Date(event.endDate);
    const endYear = endDate.getFullYear();
    const endMonth = String(endDate.getMonth() + 1).padStart(2, '0');
    const endDay = String(endDate.getDate()).padStart(2, '0');
    const endDateStr = `${endYear}${endMonth}${endDay}`;
    dtend = `DTEND;VALUE=DATE:${endDateStr}`;
  }
  
  // Handle timed events (BLS, BEA)
  let dtstart = `DTSTART;VALUE=DATE:${dateStr}`;
  if (event.time) {
    // Convert time like "08:30" to "083000" in ET timezone
    const timeStr = event.time.replace(':', '') + '00';
    dtstart = `DTSTART;TZID=America/New_York:${dateStr}T${timeStr}`;
    // Timed events are 0 duration (just marks the time)
    dtend = '';
  }
  
  return `BEGIN:VEVENT
UID:${uid}
DTSTAMP:${timestamp}
${dtstart}
${dtend}
SUMMARY:${event.title}
DESCRIPTION:${event.description}
STATUS:CONFIRMED
TRANSP:TRANSPARENT
END:VEVENT`.replace(/\n\n/g, '\n'); // Remove empty lines
}