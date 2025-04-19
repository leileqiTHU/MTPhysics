from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained('/data1/public_ckpts/gpt2/')



class MyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 1024
    def __call__(self, batch):
        prompts = [example['prompt'] for example in batch]
        completions = [' Answer: ' +example['completion'] + '[Answer Completed]' +self.tokenizer.eos_token for example in batch]
        prompts_encoded = self.tokenizer(prompts, truncation=True, padding=True, max_length=self.max_length, padding_side='left', return_tensors='pt')
        completions_encoded = self.tokenizer(completions, truncation=True, padding=True, max_length=self.max_length, padding_side='right', return_tensors='pt')
        prompt_len = prompts_encoded.input_ids.shape[-1]
        return dict(
            input_ids = torch.cat([prompts_encoded.input_ids, completions_encoded.input_ids], dim=-1),
            attention_mask = torch.cat([prompts_encoded.attention_mask, completions_encoded.attention_mask], dim=-1),
            prompt_len = prompt_len,
        )
        
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# from transformers import DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


ds_train = load_dataset('json', data_files='train_half.jsonl')['train']
data_collator = MyCollator(tokenizer=tokenizer)
dataloader = DataLoader(ds_train, batch_size=4, collate_fn=data_collator)
