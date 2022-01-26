import os
import json
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import torch
device="cuda:0"
checkpoint_n = 1500
print("Loading Linking Model")
model_location = os.path.join(os.path.dirname(__file__),"model/macbert-pipel-subtitle-chinese/checkpoint-%d"%(checkpoint_n))
tokenizer = AutoTokenizer.from_pretrained(model_location,local_files_only=True,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_location,local_files_only=True).eval().to(device)
print("Linking Model Loaded")
import urllib.request as rq
import urllib
# the first step of disambiguation might contains too much candidates, so we seperated them into different batches by mention

def mention2entity(men):
	url = "http://172.31.255.7:8982/mention/"
	data = rq.urlopen(url+urllib.parse.quote(men)).read()
	return json.loads(data)
def generatePairs(mentions):
	ret = {}
	for slice_m in mentions:
		ret[slice_m[0]] = mention2entity(slice_m[0])
	return ret
def tokenize(string,mentions):
	for i in mentions:
		keys = list(mentions[i].keys())
		header = i+"是"
		answers = [header+mentions[i][j] for j in keys]
		questions = [string]*len(keys)
		tokenized = tokenizer(questions,answers,padding=True,truncation=True,return_tensors="pt",max_length=128)
		tokenized = {k:v for k,v in tokenized.items()}
		mentions[i] = {"tokenized":tokenized,"ids":keys}
	return mentions

def get_from_model(mentions):
	for i in mentions:
		returned = model(**{k: v.to(device).unsqueeze(0) for k,v in mentions[i]["tokenized"].items()})
		logits = returned.logits[0].cpu().detach().numpy()
		mentions[i] = {mentions[i]["ids"][j]:logits[j] for j in range(len(mentions[i]["ids"]))}
	return mentions

def topkSubTitle(string,mentionScores,K=3):
	mentions = generatePairs(mentionScores)
	tokenized = tokenize(string,mentions)
	scores = get_from_model(tokenized)

	mention2scores = {i[0]:i[1] for i in mentionScores}
	for mention in scores:
		scores[mention] = [[mention,k,v+mention2scores[mention]] for k,v in scores[mention].items()]
	ret = sum(scores.values(),[])
	ret.sort(key=lambda x:x[2],reverse=True)
	return ret[:K]


