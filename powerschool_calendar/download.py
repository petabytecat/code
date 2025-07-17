from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime

# Set up Chrome options
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Uncomment to run in headless mode
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

# Initialize WebDriver
driver = webdriver.Chrome(options=options)

# Define the URL
URL = 'https://powerschool.kaust.edu.sa/guardian/home.html?_userTypeHint=student'
driver.get(URL)

try:
    # LOGIN
    email_field = WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.ID, 'identifierId'))
    )
    email_field.send_keys('dewei.zhang@thekaustschool.org')

    next_button = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, '//button[@jsname="LgbsSe" and .//span[text()="Next"]]'))
    )
    next_button.click()

    email_field = WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.NAME, 'Passwd'))
    )
    email_field.send_keys('AD!uodv423e8i_1m09')

    next_button_2 = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, '//button[@jsname="LgbsSe" and .//span[text()="Next"]]'))
    )
    next_button_2.click()

    WebDriverWait(driver, 30).until(
        EC.title_contains("Grades and Attendance")
    )

    og_startdate = datetime.datetime.strptime("08/19/2024", "%m/%d/%Y")
    og_enddate = datetime.datetime.strptime("08/25/2024", "%m/%d/%Y")
    final_enddate = datetime.datetime.strptime("06/08/2025", "%m/%d/%Y")
    dates = []

    while og_enddate <= final_enddate:
        dates.append("?startdate=" + og_startdate.strftime("%m/%d/%Y") + "&enddate=" + og_enddate.strftime("%m/%d/%Y"))
        og_startdate += datetime.timedelta(days=7)
        og_enddate += datetime.timedelta(days=7)

    events = []
    for urlend in dates:
        driver.get('https://powerschool.kaust.edu.sa/guardian/myschedule_bellsched.html' + urlend)

        WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.ID, 'tableStudentSchedMatrix'))
        )

        rows = driver.find_elements(By.ID, 'tableStudentSchedMatrix')



        for row in rows:
            event_cells = row.find_elements(By.CSS_SELECTOR, '.scheduleClass1Tick, .scheduleClass2Tick, .scheduleClass3Tick, .scheduleClass4Tick, .scheduleClass5Tick, .scheduleClass6Tick, .scheduleClass7Tick, .scheduleClass8Tick, .scheduleClass9Tick')

            for cell in event_cells:
                event_details = cell.text.strip().split("\n")
                if event_details:
                    events.append({
                        'title': event_details[0].strip(),
                        'teacher': event_details[1].strip(),
                        'room': event_details[2].strip(),
                        'time': event_details[3].strip()
                    })


    for event in events:
        print(event)
    print(len(events))


finally:
    driver.quit()

print("Scraping complete.")
