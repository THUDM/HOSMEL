import urllib.request as rq
import urllib
import json

def parse_mentions(stri):
	url = "http://172.31.255.7:8982/parse/"
	data = rq.urlopen(url+urllib.parse.quote(stri)).read()
	return json.loads(data)["data"]
