from selenium import webdriver
import time
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = 'https://dl.ibdocs.re/IB%20PAST%20PAPERS%20-%20YEAR/2015%20Examination%20Session/' 

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, dest_folder):
    local_filename = os.path.join(dest_folder, url.split('/')[-1])
    
    # Skip the file if it already exists
    if os.path.exists(local_filename):
        print(f"File already exists, skipping download: {local_filename}")
        return

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def is_directory(url):
    response = requests.head(url)
    content_type = response.headers.get('Content-Type', '')
    return 'text/html' in content_type

def parse_and_download(url, dest_folder, visited_urls, driver):
    if url in visited_urls:
        return

    visited_urls.add(url)

    print(f"Opening directory: {dest_folder}")

    driver.get(url)
    time.sleep(1)

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    links_section = soup.find('div', class_='flex flex-col gap-1')

    if links_section is None:
        print(f"No links found at {url}")
        return

    links = links_section.find_all('a', href=True)

    for link in links:
        href = link['href']
        full_url = urljoin(url, href)

        if is_directory(full_url):
            new_dest_folder = os.path.join(dest_folder, href.strip('/').split('/')[-1])
            create_directory(new_dest_folder)
            parse_and_download(full_url, new_dest_folder, visited_urls, driver)
        else:
            print(f"Downloading file: {full_url}")
            download_file(full_url, dest_folder)

# Initialize the WebDriver once
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

root_dest_folder = "downloaded_files"
create_directory(root_dest_folder)
visited_urls = set()
parse_and_download(BASE_URL, root_dest_folder, visited_urls, driver)

# Quit the WebDriver after the download is complete
driver.quit()

print("Download complete.")
