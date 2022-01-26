from transformers import AutoTokenizer, AutoModelForMultipleChoice
import urllib.request as rq
import urllib
import json
from tqdm import tqdm
import torch
import os
device="cuda:2"


model = None
tokenizer = None
checkpoint_num = 500
model_location = os.path.join(os.path.dirname(__file__),"model/macbert-hosmel-relation-chinese/checkpoint-%d"%(checkpoint_num))
print("Loading Relation Model")
tokenizer = AutoTokenizer.from_pretrained(model_location,local_files_only=True,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_location,local_files_only=True).eval().to(device)
print("Relation Model Loaded")
def getRelations(bdi):
	ret = query_neighbors_by_id_wen_new(bdi)
	ret = [i["rel"] for i in ret]
	return ret

def generatePairs(entities):
	mention_relation_pairs = []
	bdi_list = []
	for i in entities:
		relations = getRelations(i[1])
		for r in relations:
			mention_relation_pairs.append(i[0]+"的"+r)
			bdi_list.append(i[1])
	return mention_relation_pairs,bdi_list

def tokenize(q,pairs):
	qs = [q]*len(pairs)
	tokenized = tokenizer(qs,pairs,padding=True,truncation=True,return_tensors="pt",max_length=128)
	return {k:v.to(device) for k,v in tokenized.items()}

def topkRelation(q,entities,K=3):
	mention_relation_pairs,bdi_list = generatePairs(entities)
	tokenized = tokenize(q,mention_relation_pairs)
	
	returned = model(**{k:v.unsqueeze(0) for k,v in tokenized.items()})
	logits = returned.logits[0].cpu().detach().numpy()
	entity_score = {i[1]:[] for i in entities}
	for i in range(len(logits)):
		entity_score[bdi_list[i]].append((logits[i],mention_relation_pairs[i]))
	for i in entity_score:
		if len(entity_score[i]) > 0:
			entity_score[i] = max(entity_score[i],key=lambda x:x[0])
		else:
			entity_score[i] = (0,"No Relation")		
	mentions = [(entities[i][0],entities[i][1],entity_score[entities[i][1]][0]+entities[i][2],entity_score[entities[i][1]][1]) for i in range(len(entities))]
	mentions.sort(key=lambda x:x[2],reverse=True)
	return mentions[:K]
