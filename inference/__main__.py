import pandas as pd
from glob import glob
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from .utils import generate, train_sample_size
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


# df = pd.read_csv("./data/remaining_data.csv")

# inference_batch_size = 16

# model_name = "polyglot12_8B"
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-12.8b")
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/polyglot-ko-12.8b')

def main(config):
    df = pd.read_csv(config['data']['data_path'])
    balanced_df = train_sample_size(df, config['data']['sample_size_per_topic']) 
    
    pretrained_checkpoint = config['model']['pretrained_model_checkpoint']
    model = AutoModelForCausalLM.from_pretrained(pretrained_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint)
    adaptor_checkpoint = config['model']['saved_adapter_checkpoint']
    
    #
    model = PeftModel.from_pretrained(model, adaptor_checkpoint)
    model.eval()
    model.to('cuda')
    model.config.use_cache = True    

    augmented_dialogue_list = []
    batch_size = config['hyper_params']['batch_size']
    with torch.no_grad():
        for i in range(0, len(balanced_df), batch_size):
            batch_augmented_dialogues = generate(tokenizer, model, balanced_df.iloc[i:i + batch_size]['text'])
            augmented_dialogue_list.extend(batch_augmented_dialogues)
            
        balanced_df['augmented_dialogue'] = augmented_dialogue_list

    balanced_df['augmented_dialogue'] = [ item.split('### 답변:')[1] for item in balanced_df['augmented_dialogue']]


    if not os.path.exists(config['save_dir']):
        os.mkdir(config['save_dir'])
    else:
        pass    
    
    balanced_df.to_csv(f'./{config['save_dir']}/augmented.csv', index=False)    
        

if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)            