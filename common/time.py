from datetime import date, datetime

def today_():
    today = date.today()
    return today.strftime("%Y%m%d")

def now_():
    now = datetime.now()
    return now.strftime("%Y%m%d.%H:%M:%S")
