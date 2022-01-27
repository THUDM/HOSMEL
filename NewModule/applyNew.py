from transformers import AutoTokenizer, AutoModelForMultipleChoice
import urllib.request as rq
import urllib
import json
from tqdm import tqdm
import torch
import os
device="cuda:0"


model_location = os.path.join(os.path.dirname(__file__),"model")
print("Loading New Model")
tokenizer = AutoTokenizer.from_pretrained(model_location,local_files_only=True,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_location,local_files_only=True).eval().to(device)
print("New Model Loaded")

def generatePairs(entities):
	mention_new_pairs = []
	bdi_list = []
	# TODO
	for i in entities:
		relations = getRelations(i[1])
		for r in relations:
			mention_relation_pairs.append(i[0]+"的"+r)
			bdi_list.append(i[1])
	pass
	return mention_new_pairs,bdi_list

def tokenize(q,pairs):
	qs = [q]*len(pairs)
	tokenized = tokenizer(qs,pairs,padding=True,truncation=True,return_tensors="pt",max_length=128)
	return {k:v.to(device) for k,v in tokenized.items()}

def topkNew(q,entities,K=3):
	mention_new_pairs,bdi_list = generatePairs(entities)
	tokenized = tokenize(q,mention_new_pairs)
	
	returned = model(**{k:v.unsqueeze(0) for k,v in tokenized.items()})
	logits = returned.logits[0].cpu().detach().numpy()
	entity_score = {i[1]:[] for i in entities}
	for i in range(len(logits)):
		entity_score[bdi_list[i]].append((logits[i],mention_relation_pairs[i]))
	for i in entity_score:
		if len(entity_score[i]) > 0:
			entity_score[i] = max(entity_score[i],key=lambda x:x[0])
		else:
			entity_score[i] = (0,"No New")		
	mentions = [(entities[i][0],entities[i][1],entity_score[entities[i][1]][0]+entities[i][2],entity_score[entities[i][1]][1]) for i in range(len(entities))]
	mentions.sort(key=lambda x:x[2],reverse=True)
	return mentions[:K]
