import requests
from pyquery import PyQuery as pq


entry_points = [
                'http://vancouver.en.craigslist.ca/search/apa/rch?query=&srchType=A&minAsk=800&maxAsk=1500&bedrooms=&hasPic=1',
                'http://vancouver.en.craigslist.ca/search/apa/rch?hasPic=1&maxAsk=1500&minAsk=800&srchType=A&s=100'
                ]


entry_points = [
                'http://vancouver.en.craigslist.ca/apa/',
                'http://vancouver.en.craigslist.ca/apa/index100.html',
                'http://vancouver.en.craigslist.ca/apa/index200.html',
                'http://vancouver.en.craigslist.ca/apa/index300.html'
                ]


ads = []


for url in entry_points:
    html = requests.get(url).content
    selector = pq(html)
    ads.extend([ n.get('href') for n in selector('.row a') ])
    

for url in ads:
    #print "'" + url + "',"
    print url