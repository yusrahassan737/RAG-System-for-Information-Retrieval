# code to parse individual report
from bs4 import BeautifulSoup
import os

all_reports = os.listdir("reports")
for report in all_reports:
    with open("reports/" + report, "r", encoding = "utf-8") as f:
        text = f.read()

    soup = BeautifulSoup(text, "html.parser")

    add_to_file = [soup.find('meta', attrs={'name': 'dcterms.title'}),
                soup.find('meta', attrs={'name': 'dcterms.creator'}),
                soup.find('meta', attrs={'name': 'dcterms.modified'}),
                soup.find('title'),
                soup.find('article')]
    open_file = open("test.txt", "w", encoding = "utf-8")
    for i in add_to_file:
        if i:
            open_file.write(i.text + "\n")

    open_file.close()