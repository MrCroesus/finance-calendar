import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from bs4 import BeautifulSoup
from urllib.request import urlopen

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def main():
  """Shows basic usage of the Google Calendar API.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    service = build("calendar", "v3", credentials=creds)
    return service

  except HttpError as error:
    print(f"An error occurred: {error}")


def month_to_str(abbreviation):
  match abbreviation[:3]:
    case "Jan":
      return '01'
    case "Feb":
      return '02'
    case "Mar":
      return '03'
    case "Apr":
      return '04'
    case "May":
      return '05'
    case "Jun":
      return '06'
    case "Jul":
      return '07'
    case "Aug":
      return '08'
    case "Sep":
      return '09'
    case "Oct":
      return '10'
    case "Nov":
      return '11'
    case "Dec":
      return '12'


def process_FOMC_meeting(year, dateTime):
  # get starred
  starred = False
  if '*' in dateTime:
    starred = True
    dateTime = dateTime.replace('*', '')

  # parse string
  month = ""
  day = ""

  for i in range(len(dateTime)):
    if dateTime[i].isnumeric():
      month = dateTime[:i]
      day = dateTime[i:]
      break

  # parse month
  start_month = ""
  end_month = ""

  if '/' in month:
    months = month.split('/')
    start_month = month_to_str(months[0])
    end_month = month_to_str(months[1])
  else:
    start_month = month_to_str(month)
    end_month = month_to_str(month)

  # parse day
  days = day.split('-')
  start_day = '0' + days[0] if len(days[0]) == 1 else days[0]
  end_day = '0' + days[1] if len(days[1]) == 1 else days[1]

  return starred, year + '-' + start_month + '-' + start_day + 'T00:00:00-05:00', year + '-' + end_month + '-' + end_day + 'T23:59:59-05:00'
  

def get_dates(tables, start_row, date_cell):
    days = []
    times = []
    day_or_time = True

    for table in tables:
        if len(table) > 1:
            rows = table.find_all('tr')
            
            selected_rows = rows[start_row:]
            
            i = 0

            while i in range(len(selected_rows)):
                cells = selected_rows[i].find_all('td')

                if len(cells) == 0:
                    return

                date = cells[date_cell].text.strip()
                                
                if 'Updated' in date:
                    i += 2
                    continue
                
                if day_or_time:
                    days.append(date)
                else:
                    times.append(date)
                
                day_or_time = not day_or_time
                i += 1
    
    return days, times
  
  
def process_BEA_release_date(date, time):
  # parse date
  _, month, day, year = date.split(' ')
  
  # process month and day
  month = month_to_str(month[:3])
  day = day[:-1]
  
  # parse time
  clock_time, AMorPM = time.split(' ')
  
  # process hour and minutes
  hour, minute = clock_time.split(':')
  hour = str(int(hour) + 12) if AMorPM == 'pm' else hour
  minute = '0' + minute if len(minute) == 1 else minute

  return year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':00-06:00', year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':00-06:00'
  
  
def dateTime_equals(dateTime1, dateTime2):
  # parse dates and Times
  date1, Time1 = dateTime1.split('T')
  date2, Time2 = dateTime2.split('T')
  
  # parse years, months, and days
  year1, month1, day1 = date1.split('-')
  year2, month2, day2 = date2.split('-')
  
  # parse timezones and times
  timezone1, time1 = Time1.split('-')
  timezone2, time2 = Time2.split('-')
  

def new_event(service, summary, description, start_dateTime, end_dateTime):
  event = {
    'summary': summary,
    'description': description,
    'start': {
      'dateTime': start_dateTime,
      'timeZone': 'America/Los_Angeles',
    },
    'end': {
      'dateTime': end_dateTime,
      'timeZone': 'America/Los_Angeles',
    },
    'reminders': {
      'useDefault': True,
    },
  }

  event = service.events().insert(calendarId='c_f33c38db2bbb06fd8e46447a560fc94d9d212b70c9dea85a75092342cff9b046@group.calendar.google.com', body=event).execute()
  print('Event created: %s' % (event.get('htmlLink')))


if __name__ == "__main__":
  service = main()
  
  finance_events_result = (
    service.events()
    .list(
        calendarId="c_f33c38db2bbb06fd8e46447a560fc94d9d212b70c9dea85a75092342cff9b046@group.calendar.google.com",
        timeMin="2024-01-01T00:00:00Z",
        timeMax="2024-12-31T23:59:59Z",
        maxResults=2500,
        timeZone="-05:00",
        singleEvents=True,
        orderBy="startTime",
    )
    .execute()
  )
  finance_events = finance_events_result.get("items", [])
  finance_event_starts = [finance_event["start"]["dateTime"] for finance_event in finance_events]

  # BLS (unemployment and inflation) info is added by URL: https://www.bls.gov/schedule/news_release/bls.ics
  # located: https://www.bls.gov/schedule/news_release/cpi.htm

  # FOMC (interest rate) info is added manually: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
  FOMC_meetings = ["January30-31",
  "March19-20*",
  "Apr/May30-1",
  "June11-12*",
  "July30-31",
  "September17-18*",
  "November6-7",
  "December17-18*"]

  for FOMC_meeting in FOMC_meetings:
    starred, start_dateTime, end_dateTime = process_FOMC_meeting('2024', FOMC_meeting)

    if start_dateTime not in finance_event_starts:
      summary = "FOMC Meeting"
      description =  "Interest rate changes & Summary of Economic Projections" if starred else "Interest rate changes"

      new_event(service, summary, description, start_dateTime, end_dateTime)

  # BEA (GDP) info is scraped: https://fred.stlouisfed.org/releases/calendar?od=asc&rid=53&ve=2020-12-31&view=year&vs=2020-01-01
  gdp_dates_url = urlopen('https://fred.stlouisfed.org/releases/calendar?od=asc&rid=53&ve=2020-12-31&view=year&vs=2020-01-01')
  gdp_dates = BeautifulSoup(gdp_dates_url, 'html.parser')
  gdp_dates_table = gdp_dates.find_all('table')
  days, times = get_dates(gdp_dates_table, 1, 0)
  for day, time in zip(days, times):
    start_dateTime, end_dateTime = process_BEA_release_date(day, time)

    if start_dateTime not in finance_event_starts:
      summary = "BEA Release Date"
      description =  "GDP estimates released"

      new_event(service, summary, description, start_dateTime, end_dateTime)
