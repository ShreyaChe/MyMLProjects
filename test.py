import requests
from bs4 import BeautifulSoup
from datetime import date
import calendar

def news():
    # the target we want to open
    url = 'https://www.????'

    # open with GET method
    resp = requests.get(url)

    # http_respone 200 means OK status
    if resp.status_code == 200:
        print("Successfully opened the web page")
        print("The news are as follow :-\n")
        my_date = date.today()
        tst = calendar.day_name[my_date.weekday()]

        # we need a parser,Python built-in HTML parser is enough .
        soup = BeautifulSoup(resp.text, 'html.parser')

        # l is the list which contains all the text i.e news
        l = soup.findAll("h3", {"class": "product-name"})
       
        # now we want to print only the text part of the anchor.
        # find all the elements of a, i.e anchor
        for j in l: 
          for i in j.contents:
            f = open("demofile.txt", "a",encoding='utf-8')
            f.write(tst+'-'+i+'\n')
            print(i)
        for i in range (20):
            f.write('*')
        f.write('\n')
    else:
        print("Error")


news()

