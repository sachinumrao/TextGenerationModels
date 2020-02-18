import requests
import time
from bs4 import BeautifulSoup as bs

base_url = "https://fangj.github.io/friends/"

page = requests.get(base_url)
assert page.status_code == 200

soup = bs(page.content, 'html.parser')
links = soup.find_all('a')

useful_links = []

# Extract links
for link in links:
    try:
        url_suffix = link["href"]
        useful_links.append(url_suffix)
    except KeyError:
        pass

# Extract content
t1 = time.time()
data_file = open("../../../Data/LM/friends_script.txt", "w")
for eps in useful_links:
    eps_url = base_url+eps
    print(eps_url)
    eps_page = requests.get(eps_url)
    assert eps_page.status_code == 200
    eps_soup = bs(eps_page.content, 'html.parser')
    eps_ptags = eps_soup.find_all('p')
    for tag in eps_ptags:
        text = tag.getText()+'\n'
        data_file.write(text)

data_file.close()
t2 = time.time()
print("Time Taken: ", t2-t1)