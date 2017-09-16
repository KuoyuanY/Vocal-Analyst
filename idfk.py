import requests
from bs4 import BeautifulSoup
song_name = "Kings Never Die"
artist_name = "Eminem"


link = "https://www.youtube.com/results?search_query=" + song_name + ' - ' + artist_name
f = requests.get(link)
soup = BeautifulSoup(f, 'html.parser')
soup = soup.prettify()
line = soup.find(string="watch?")

print line