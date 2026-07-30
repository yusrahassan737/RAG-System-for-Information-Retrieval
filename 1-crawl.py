# Date: 2026/07/27
# Class: CP423
# Description: Script to extract documents from the tsb website

from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import os

# Access root page
root_url = "https://tsb.gc.ca"
main_url = root_url + "/eng/rapports-reports/rail/index.html"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"} # helps request get accepted by website
time.sleep(2)
root_page = requests.get(main_url, headers=headers)

# Parse with beautiful soup
soup = BeautifulSoup(root_page.text, "html.parser")
table = soup.tbody

# Extract all page links
links = table.find_all("a", href=True)
doc_info = {"doc_id": [], "source": [], "investigation_num": []}
for i in range(len(links)):
    full_link = root_url + links[i].get("href").strip()
    doc_info["doc_id"].append(i)
    doc_info["source"].append(full_link)
    doc_info["investigation_num"].append(str(links[i].text))

# Visit each link and download page
for i in range(len(links)):
    investigation_num = doc_info["investigation_num"][i]
    if (investigation_num + ".html") not in os.listdir("reports"):
        time.sleep(2)
        report = requests.get(doc_info["source"][i], headers=headers)
        with open("reports/" + investigation_num + ".html", "w", encoding="utf-8") as f:
            f.write(report.text)
    else:
        print(f"{investigation_num}.html is already in the folder")

# Save links
doc_info_df = pd.DataFrame(doc_info)
doc_info_df.to_csv("doc_info.csv", index = False)

print(f"There are {i + 1} reports downloaded.")