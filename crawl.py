# Name: Yusra Hassan
# Date: 2026/07/27
# Class: CP423
# Description: Script to extract documents from the tsb website

from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

# Access root page
root_url = "https://tsb.gc.ca"
main_url = root_url + "/eng/rapports-reports/rail/index.html"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"} # apparently helps request get accepted
time.sleep(2)
root_page = requests.get(main_url, headers=headers)

# Parse with beautiful soup
soup = BeautifulSoup(root_page.text, "html.parser")
table = soup.tbody

# Extract all page links
links = table.find_all("a", href=True)
links_tidy = {}
for i in links:
    full_link = root_url + i.get("href").strip()
    links_tidy[str(i.text)] = full_link

# Visit each link and download page
for investigation_num in links_tidy:
    time.sleep(2)
    report = requests.get(links_tidy[investigation_num], headers=headers)
    with open("reports/" + investigation_num + ".txt", "w") as f:
        f.write(report.text)

# Save links
with open("page_contents.txt", "w") as f:
    f.write(str(links_tidy))
