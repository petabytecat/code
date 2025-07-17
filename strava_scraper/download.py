from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import datetime
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, NoSuchElementException

options = webdriver.ChromeOptions()
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

driver = webdriver.Chrome(options=options)

URL = 'https://www.strava.com/login'
driver.get(URL)

# Wait for the Google sign-in button to be present and clickable
wait = WebDriverWait(driver, 10)
google_sign_in_button = wait.until(
    EC.element_to_be_clickable((By.ID, "google-signin"))
)
google_sign_in_button.click()

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

time.sleep(2)

club_urls = ['https://www.strava.com/clubs/163317', "https://www.strava.com/clubs/kaustcyclingforfun", "https://www.strava.com/clubs/KAUST", "https://www.strava.com/clubs/KAUSTVV"]
athlete_links = set()

# First, collect all athlete links from all clubs
for club_url in club_urls:
    driver.get(club_url)
    time.sleep(1)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    for link in soup.find_all("a", class_="avatar-content"):
        profile_url = link.get("href")
        if profile_url and "/athletes/" in profile_url:
            athlete_links.add(profile_url)

    time.sleep(1)

# Then, process each athlete's profile
for athlete_link in athlete_links:
    try:
        full_url = f'https://www.strava.com{athlete_link}'
        driver.get(full_url)
        time.sleep(2)

        # Try to find and click the ride button
        try:
            wait = WebDriverWait(driver, 5)
            ride_button = wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "span.app-icon.icon-ride.icon-lg"))
            )
            ride_button.click()
            time.sleep(2)  # Wait for content to load after clicking

            # Get the updated page content after clicking
            athlete_html = driver.page_source
            athlete_soup = BeautifulSoup(athlete_html, 'html.parser')

            # Find and process the 20K time
            rows = athlete_soup.find_all('tr')
            for row in rows:
                cell = row.find('td')
                if cell and cell.text.strip() == '20K':
                    time_cell = row.find_all('td')[1]
                    if time_cell:
                        time_text = time_cell.text.strip()
                        print(f"Athlete {athlete_link}: {time_text}")

        except (TimeoutException, NoSuchElementException):
            print(f"Ride button not found for {athlete_link}, skipping to next athlete")
            continue

    except Exception as e:
        print(f"Error processing athlete {athlete_link}: {str(e)}")
        continue

driver.quit()
