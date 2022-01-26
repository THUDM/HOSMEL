import ahocorasick
import pickle
from flask import Flask
from tqdm import tqdm
import json
import os
app = Flask(__name__)
app.config['JSON_AS_ASCII'] =False

print("Loading Entity List")
AC_machine = ahocorasick.load(os.path.join(os.path.dirname(__file__),'nameTri'),pickle.loads)
subList = json.load(open(os.path.join(os.path.dirname(__file__),'subList.json'),'r'))
print("Preprocessing Entity List")
for i in tqdm(range(len(subList))):
	subList[i] = {j[1]:j[0] for j in subList[i]}
print("Finished Preprocess Entity List")

print("Loading bdi2relation")
with open('bdi2relation.pkl','rb') as f:
	bdi2relation = pickle.load(f)
print("Finished bdi2relation")

@app.route("/parse/<string:stri>")
def parse_mentions(stri):
	return {"data":list(set([i[1][1] for i in AC_machine.iter(stri.lower())]))}

@app.route("/mention/<string:men>")
def mention2id(men):
	return subList[AC_machine.get(men)[0]]

@app.route("/relation/<string:bdi>")
def getRelation(bdi):
	data = {'data': bdi2relation.get(bdi,[])}
	return data

app.run(port=8982,host="0.0.0.0")