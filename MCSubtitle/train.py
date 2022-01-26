from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import pickle
from datasets import load_dataset,concatenate_datasets
batch_size = 16
log_rate = 10
model_name = "hfl/chinese-macbert-large"
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)
model = AutoModelForMultipleChoice.from_pretrained(model_name)
encoded_data = load_dataset("json",data_files={"train":'./processedData.json'},cache_dir="../cache/")
encoded_data = encoded_data["train"]
encoded_data = concatenate_datasets([encoded_data.shard(num_shards=20,index=i) for i in [16,17,18,19]])
args = TrainingArguments(
	output_dir="./model/macbert-hosmel-subtitle-chinese",
	learning_rate=1e-5,
	per_device_train_batch_size=batch_size,
	num_train_epochs=3,
	weight_decay=1e-3,
)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
	tokenizer:PreTrainedTokenizerBase
	padding:Union[bool, str, PaddingStrategy] = True
	max_length: Optional[int]=None
	pad_to_multiple_of:Optional[int] = None
	
	def __call__(self, features):
		label_name = "label"
		labels = [feature.pop(label_name) for feature in features]
		batch_size = len(features)
		num_choices = 4
		flattened_features = [[{k:v[i] for k,v in feature.items()} for i in range(num_choices)]for feature in features]
		flattened_features = sum(flattened_features,[])
		batch = self.tokenizer.pad(
			flattened_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt"
		)
		batch = {k:v.view(batch_size,num_choices,-1) for k,v in batch.items()}
		batch["labels"] = torch.tensor(labels,dtype=torch.int64)
		return batch

import numpy as np

def compute_metrics(eval_predictions):
	predictions,label_ids=eval_predictions
	preds = np.argmax(predictions,axis=1)
	return {"accuracy":(preds==label).astype(np.float32).mean().item()}
import torch
print(isinstance(encoded_data,torch.utils.data.IterableDataset))	
print(len(encoded_data))
trainer = Trainer(
	model,
	args,
	train_dataset=encoded_data,
	tokenizer=tokenizer,
	data_collator=DataCollatorForMultipleChoice(tokenizer),
	compute_metrics=compute_metrics,
)
print(len(trainer.get_train_dataloader().dataset))
print(trainer.train_dataset[0])

trainer.train()

