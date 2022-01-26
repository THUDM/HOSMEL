from transformers import AutoTokenizer, AutoModelForMultipleChoice
from tqdm import tqdm
import torch
import os
device="cuda:2"


model = None
tokenizer = None
checkpoint_num = 1500
# model_location = os.path.join(os.path.dirname(__file__),"model/macbert-linking-chinese-baidu-add-alias-v4-lr_1e-5-wd_1e-3-eligant_data4_20/checkpoint-%d"%(checkpoint_num))
model_location = os.path.join(os.path.dirname(__file__),"model/macbert-hosmel-mention-chinese/checkpoint-%d"%(checkpoint_num))

print("Loading Mention Model")
tokenizer = AutoTokenizer.from_pretrained(model_location,local_files_only=True,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_location,local_files_only=True).eval().to(device)
print("Mention Model Loaded")

def topkMention(q,mentions,K=3):
	questions = [q]*len(mentions)
	tokenized = tokenizer(questions,mentions,padding=True,truncation=True,return_tensors="pt",max_length=128)
	tokenized = {k:v.to(device) for k,v in tokenized.items()}
	
	returned = model(**{k:v.unsqueeze(0) for k,v in tokenized.items()})
	logits = returned.logits[0].cpu().detach().numpy()
	mentions = [(mentions[i],logits[i]) for i in range(len(mentions))]
	mentions.sort(key=lambda x:x[1],reverse=True)
	return mentions[:K]

def filterMention(mentionDetection,mentions):
	return {k[0]:mentions[k[0]] for k in mentionDetection}
