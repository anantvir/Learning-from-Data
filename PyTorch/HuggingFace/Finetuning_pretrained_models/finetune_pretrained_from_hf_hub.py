"""
Author :  Anantvir Singh

This is straight for HuggingFace NLP course. This is good intro to HF Fine tuning. There are 2 ways to fine tune

1. Use HF Trainer API (We do not use that here)
2. Write custom training and eval loop in PyTorch (We follow this approach here)

HuggingFace provides very easy to use wrappers on top of all models that they provide on thei model hub. The only part to pay attention
to is dataset prepration using laod_dataset. You can even load custom datasets locally or remotely using load_dataset.
After that tokenize your dataset and feed it into PyTorch DataLoader. Let HF do rest of the heavy lifting.

Reference : https://huggingface.co/learn/nlp-course/chapter3/4?fw=pt
"""

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch

# ------------------------------------------------------- Load dataset from hub -------------------------------
raw_datasets = load_dataset("glue", "mrpc")

# ------------------------------------------------------- preprocess dataset -------------------------------

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True) # For very key in raw_datasets DatasetDict, apply tokenize_fn
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # just takes tokenizer as input and knows how to put batches together and what type of padding does this model require


# remov and rename some columns
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1","sentence2","idx"])
tokenized_datasets = tokenized_datasets.rename_column("label","labels")
#print(tokenized_datasets)

tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)     # These are all the columns our model needs

# ----------------------------- Define PyTorch DataLoader --------------------------------------

from torch.utils.data import DataLoader

"""
https://pytorch.org/docs/stable/data.html
PyTorch DataLoader takes as input an implementation of Dataset. It support 2 types of implementations of Dataset
1. Map style implementation (implementing __getitem()__ and __len()__)
2. Iterable style (implementing __iter__())
"""
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=True, batch_size=8, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2) 
# We can add num_labels field in from_pretrained in Bert model. Refer : https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/bert#transformers.BertForSequenceClassification

optimizer = AdamW(model.parameters(), lr = 5e-5)


# ------------------------------- Configure learning rate scheduler ------------------------------------
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))  # configure progress bar
 
# ------------------------------------------------- Training Loop --------------------------------------

model.train()           # Set the model to train mode. This will activate some layers example dropout

for epoch in range(num_epochs):
    for batch in train_dataloader:
        labels = batch["labels"]
        input_tensor = batch["input_ids"]
        batch = {k : v.to(device) for k,v in batch.items()}
        """{'attention_mask': torch.Size([8, 65]),
            'input_ids': torch.Size([8, 65]),
            'labels': torch.Size([8]),
            'token_type_ids': torch.Size([8, 65])}"""

        outputs = model(**batch)

        loss = outputs.loss # Calculate loss wrt all model parameters
        loss.backward()     # Calculate gradients

        optimizer.step()    # Update weights
        lr_scheduler.step()  # Weight decay step
        optimizer.zero_grad() # Zero gradients for next batch
        progress_bar.update(1)



# ----------------------------------------- Evaluation Loop --------------------------------------
import evaluate

metric = evaluate.load("glue", "mrpc")  # Load standard evaluation metric for mrpc dataset in glue benchmarks

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)
    metric.add_batch(predictions=predictions, references=batch["labels"]) # accumulate metrics for each batch and then get final result in end after accumulating for all batches

print("Evaluation Results :", metric.compute())




