import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

class GAPMLMDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['Text'].lower()

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
    df_gap = pd.read_csv("gap_flipped.tsv", delimiter="\t")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
    model.to(device)

    # train-validation split (80%-20%)
    train_data, eval_data = train_test_split(df_gap, test_size=0.2, random_state=42)

    train_dataset = GAPMLMDataset(train_data, tokenizer)
    eval_dataset = GAPMLMDataset(eval_data, tokenizer)

    # data collator for dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # enable MLM objective
        mlm_probability=0.15,  # dynamically mask 15% of tokens
        return_tensors="pt"
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=2e-5,  # fine-tuning learning rate (1e-5 to 3e-5)
        per_device_train_batch_size=16,  
        per_device_eval_batch_size=32,  
        num_train_epochs=3,
        weight_decay=0.01,  
        warmup_steps=500,  
        evaluation_strategy='epoch',  # evaluate after every epoch
        save_strategy='epoch',  # save at the end of every epoch
        logging_dir='./logs',
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
        
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # learning rate
        betas=(0.9, 0.98),  # betas from RoBERTa paper
        weight_decay=0.01,  # regularization from RoBERTa paper (weight decay)
    )

    # learning rate scheduler with polynomial decay
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,  # warmup steps for fine-tuning
        num_training_steps=training_args.max_steps, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
        data_collator=data_collator,  # use the dynamic masking data collator
        tokenizer=tokenizer,
        optimizers=(optimizer, scheduler)  # custom optimizer and scheduler
    )

    trainer.train()

    model.save_pretrained("./finetunedRL")
    tokenizer.save_pretrained("./finetunedRL")
    