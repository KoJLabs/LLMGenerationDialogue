import pandas as pd
from .utils import train_sample_size
from .model import load_tokenizer_model
import argparse
from datasets import Dataset
import torch
import math
import transformers
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="config.yaml",help="config path")
    args = parser.parse_args()
    return args

def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    return config



def main(config):
    torch.cuda.empty_cache()
    
    sample_size_per_topic = config['params']['sample_size_per_topic']
    data_path = config['data']['data_path']
    
    df = pd.read_csv(data_path)
    
    balanced_df = train_sample_size(df, sample_size_per_topic) 
    balanced_df = balanced_df.sample(frac = 1, random_state=42)
    balanced_df = balanced_df.reset_index(drop=True)

    data = Dataset.from_pandas(balanced_df)
    
    #
    tokenizer, model = load_tokenizer_model(config['model']['checkpoint'])
    model.config.use_cache = False
    
    data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
        
    # Set the number of epochs
    num_epochs = config['model']['hyperparams']['epochs']
    batch = config['model']['hyperparams']['batch_size']
    learning_rate_score = config['model']['hyperparams']['learning_rate']
    
    total_train_steps = math.ceil((len(balanced_df) / batch) * num_epochs)
    print(total_train_steps)    
    
    tokenizer.pad_token = tokenizer.eos_token
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=1,
            lr_scheduler_type = "linear",
            learning_rate= learning_rate_score,
            fp16=True,
            logging_steps=10,
            output_dir=f"{config['model']['save_model_path']}/8bit_outputs_{sample_size_per_topic}",
            optim="paged_adamw_8bit",
            save_total_limit = 3, #add
            save_strategy = "epoch", #add
            max_steps=total_train_steps,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )    
    
    trainer.train()

    
if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)    