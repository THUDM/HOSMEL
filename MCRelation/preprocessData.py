from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import pickle
from datasets import load_dataset
batch_size = 32
log_rate = 10
model_name = "hfl/chinese-macbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

data = load_dataset("json",data_files={"train":'./data/extract_training.json'},cache_dir="../cache/")
data = data["train"]
ending_names = ["target0","target1","target2","target3"]
def preprocess(example):
	first_sentences = [[context]*4 for context in example["sentence"]]
	question_headers = ["" for mention in example["mention"]]
	second_sentences = [[header+example[end][i] for end in ending_names] for i,header in enumerate(question_headers)]
	first_sentences = sum(first_sentences,[])
	second_sentences = sum(second_sentences,[])
	tokenized = tokenizer(first_sentences,second_sentences,truncation=True,max_length=128)
	tokenized = {k:[v[i:i+4] for i in range(0, len(v), 4)] for k,v in tokenized.items()}
	return tokenized

def preprocess_no_batch(example):
	first_sentences = [example["sentence"]]*4
	question_header = example["mention"]+"是"
	second_sentences = [question_header+example[end] for end in ending_names]
	tokenized = tokenizer(first_sentences,second_sentences,truncation=True,max_length=128)
	tokenized["label"] = example["Label"]
	return tokenized
c_names = data.column_names
c_names.remove("Label")
encoded_data = data.map(preprocess,batched=True,remove_columns=c_names,num_proc=4).rename_column("Label","label")
print(encoded_data[0])
print(data[0])
encoded_data.to_json("./processedData.json")
