# code to parse individual report
from bs4 import BeautifulSoup

with open("test.html", "r") as f:
    text = f.read()

soup = BeautifulSoup(text, "html.parser")


add_to_file = [soup.find('meta', attrs={'name': 'dcterms.title'}),
            soup.find('meta', attrs={'name': 'dcterms.creator'}),
            soup.find('meta', attrs={'name': 'dcterms.modified'}),
            soup.find('title'),
            soup.find('h1', attrs={'class': 'page-header'})]
open_file = open("test.txt", "w")
for i in add_to_file:
    if i.text:
        open_file.write(i.text + "\n")

open_file.close()

soup.find_all()


# things to keep
"""
<meta name = dcterms.title
dcterms.creator
dcterms.modified
<title>
<h1 id="wb-cont" class="page-header mrgn-tp-md">Railway Investigation Report R95Q0019</h1>
everything-ish inside of article
"""