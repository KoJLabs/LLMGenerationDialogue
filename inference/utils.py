import pandas as pd
import torch
import numpy as np

def train_sample_size(df, sample_size):
    label_counts = df['single_topic'].value_counts()
    balanced_df = pd.DataFrame(columns=df.columns)
    for label, count in label_counts.items():
        # Select up to 200 rows for each label
        label_subset = df[df['single_topic'] == label].sample(min(sample_size, count), random_state=42)
        balanced_df = pd.concat([balanced_df, label_subset])
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df


def generate(tokenizer_, model_, texts):
    torch.cuda.empty_cache()
    inputs_encoded = tokenizer_(
        [text.split('답변:')[0] + '답변:' for text in texts],
        padding=True,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=False,
    )
    inputs_encoded = {key: value.to('cuda') for key, value in inputs_encoded.items()}
    
    gened = model_.generate(
        **inputs_encoded,
        top_p = 0.5, # 0.75
        max_new_tokens = 256,
        early_stopping=True,
        do_sample=True,
        no_repeat_ngram_size=2,
        eos_token_id=2,
        )
    
    decoded_outputs = tokenizer_.batch_decode(gened, skip_special_tokens=True)

    if np.random.choice([0,1,2,3]) == 3:
        print(decoded_outputs[0])

    return decoded_outputs