from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator

from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from datasets import load_dataset
tqdm.pandas()

dataset_name = "lvwerra/stack-exchange-paired"
#reward_model_name = "cambioml/rlhf-reward-model"
reward_model_name = "weqweasdas/hh_rlhf_rm_open_llama_3b"
rl_model_name = "openai-community/gpt2"
reward_baseline = 0.3 # randomly initialized 

# Create configuration for PPO Trainer
config = PPOConfig(
    steps=10,
    model_name=rl_model_name,
    learning_rate=1e-6,
    optimize_cuda_cache=True,
    early_stopping=True,
    ppo_epochs= 3
)
train_dataset = load_dataset(dataset_name, split="train", verification_mode="no_checks")
train_dataset = train_dataset.select(range(1000))
original_columns = train_dataset.column_names

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=rl_model_name)

# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

"""
This function will take a batch of :
                                    Dataset({
                                        features: ['qid', 'question', 'date', 'metadata', 'response_j', 'response_k'],
                                        num_rows: 100000
                                    })
and then create a prompt ready to be fed into the RL_LLM. Prompt will be in the format

input_prompt : "Question : <user query>
 
 Answer :
"
We then tokenize input_prompt (query in this case) and we store the dataset in following format
        Dataset({
            features: ['query', 'input_ids'],
            num_rows: 100000
        })
"""
def build_dataset(tokenizer, dataset_name):  
    def preprocess_fn(examples):
        new_dataset = {
            "user_query_string" : [],
            "input_ids" : []
        }
        for question in examples["question"]:
            query = "Question :" + question + " \n\n Answer : "
            tokenized_query = tokenizer(query, truncation = True)
            new_dataset["user_query_string"].append(query)
            new_dataset["input_ids"].append(tokenized_query["input_ids"])

        return new_dataset
    
    ds = train_dataset.map(
        preprocess_fn,
        batched = True,
        remove_columns = original_columns
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(tokenizer, dataset_name)

lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    rl_model_name,
    peft_config=lora_config,
)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    data_collator=collator,
    dataset=dataset,
)

# current_device = Accelerator.local_process_index
# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 0 if torch.cuda.is_available() else "cpu"
device = 0 if torch.cuda.is_available() else -1
print(torch.device)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    tokenizer=tokenizer,
    return_token_type_ids=False,
    device=device
)

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = sentiment_pipe.model.config.eos_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}

output_min_length = 32
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

    # When a Huggingface Dataset is passed to PPOTrainer, PPOTrainer only keeps the columns
    # that are used by the model and removes all other columns e.g in this case we it only keeps
    # the column "input_ids" and removes the column "user_query_string". Shape of question tensors is (batch_size * len_of_questions)
    question_tensors = batch["input_ids"] # Shape : (batch_size * question_len) Data is unpadded here. It will be padded when ppotrainer is called
    
    # Convert question ids to strings and store them in a key in batch dictionary to be used later
    batch["query_text"] = tokenizer.batch_decode(question_tensors)

    # Shape of response : (batch_size * generation_length_of_each_response)
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt = False,
        **generation_kwargs,
        length_sampler=output_length_sampler,
        batch_size = 1
    )
    # Convert response ids to sentences
    batch["response_text"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute reward score for entire batch of generations using sentiment pipeline (since we are predicting a scalar value just like sentiment)
    query_response_text_combined = [ q + r for q,r in zip(batch["query_text"], batch["response_text"])]
    pipeline_outputs = sentiment_pipe(query_response_text_combined, **sent_kwargs)

    # Shape : (batch_size * 1) -> We get a scalar output for each example in batch
    rewards = [torch.tensor(output[0]["score"] - reward_baseline) for output in pipeline_outputs]

    # Run PPO step
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

    # https://huggingface.co/docs/transformers/en/main_classes/tokenizer (log_stats function)
    ppo_trainer.log_stats(stats, batch, rewards)
    print("Finishing epoch :", epoch)
    break