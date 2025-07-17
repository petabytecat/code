import os
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import csv
import re

base_url = "https://materovcompetition.org/{}-competition-archive"
years = reversed(list(range(2018, 2025)))
urls = [[year, base_url.format(year)] for year in years]

output_folder = "spec_sheets_pngs"
os.makedirs(output_folder, exist_ok=True)

# Function to read CSV file and create a dictionary of team names and rankings
def read_scoresheet(year):
    team_rankings = {}
    csv_filename = f'{year}_scoresheet.csv'
    try:
        with open(csv_filename, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader)  # Skip the header row

            for row in csvreader:
                if len(row) >= 2:
                    team_name = row[0].strip()
                    try:
                        ranking = int(row[1])
                        team_name_key = team_name.replace(' ', '_')
                        team_rankings[team_name] = ranking
                        team_rankings[team_name_key] = ranking
                    except ValueError:
                        print(f"Invalid ranking value for team {team_name} in {csv_filename}")

        print(f"Teams and rankings found in {csv_filename}:")
        for team, rank in team_rankings.items():
            print(f"{team}: {rank}")

    except FileNotFoundError:
        print(f"Scoresheet for year {year} not found.")
    return team_rankings

def download_pdf(pdf_url, output_path):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {output_path}")
        return True
    else:
        print(f"Failed to download: {pdf_url}")
        return False

def convert_pdf_to_png(pdf_path, output_folder):
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    doc = fitz.Document(pdf_path)
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi = 300)
        output_image_path = os.path.join(output_folder, f"{os.path.basename(pdf_path)[:-4]}_page{page_num + 1}.png")
        pix.save(output_image_path)
        print(f"Converted: {output_image_path}")
    doc.close()

    os.remove(pdf_path)
    print(f"Deleted: {pdf_path}")

def extract_team_keyword(full_name):
    # Split by dash if present, then by space, and take the first non-empty part
    parts = re.split(r'[-\s]+', full_name.strip())
    return next((part for part in parts if part), '')

for year, url in urls:
    team_rankings = read_scoresheet(year)

    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    for element in soup.find_all(['h2', 'h3', 'li']):
        if element.name in ['h2', 'h3']:
            current_team_category = element.text.strip()

        if element.name == 'li':
            a_tags = element.find_all('a')
            for a in a_tags:
                if "Spec Sheet" in a.text:
                    spec_sheet_url = a['href']
                    full_team_name = element.text.split("[")[0].strip()
                    category_name = current_team_category.replace(' ', '_')

                    if full_team_name != "Company Spec Sheet" and category_name == "RANGER":
                        # Extract the team name, ignoring the (RNxx) part
                        team_name_parts = full_team_name.split(' - ')
                        if len(team_name_parts) > 1:
                            team_name = ' '.join(team_name_parts[1:])
                        else:
                            team_name = team_name_parts[0]

                        # Remove the (RNxx) part if present
                        team_name = re.sub(r'\s*\([Rr][Nn]\d+\)\s*$', '', team_name).strip()

                        # Take the first word of the team name for matching, or use full name if it's a single word
                        team_name_key = team_name.split()[0] if team_name else full_team_name

                        matching_team = next((team for team in team_rankings.keys()
                                              if team.lower().startswith(team_name_key.lower())), None)

                        if matching_team:
                            ranking = team_rankings[matching_team]
                        else:
                            ranking = 0
                            print(f"Warning: No ranking found for team {full_team_name}")

                        spec_sheet_name = f"{year}_{category_name}_{full_team_name.replace(' ', '_')}_Spec_Sheet_rank{ranking:02d}.pdf"
                        pdf_output_path = os.path.join(output_folder, spec_sheet_name)

                        print(f"Processing team: {full_team_name}")
                        print(f"Extracted team name: {team_name}")
                        print(f"Team name key for matching: {team_name_key}")
                        print(f"Matching team found: {matching_team}")
                        print(f"Ranking found: {ranking}")

                        if os.path.exists(pdf_output_path):
                            print(f"File already exists, skipping: {pdf_output_path}")
                            continue

                        if download_pdf(spec_sheet_url, pdf_output_path):
                            convert_pdf_to_png(pdf_output_path, output_folder)
