import pandas as pd

def train_sample_size(df, sample_size):
    label_counts = df['single_topic'].value_counts()
    balanced_df = pd.DataFrame(columns=df.columns)
    
    for label, count in label_counts.items():
        label_subset = df[df['single_topic'] == label].sample(min(sample_size, count), random_state=42)
        balanced_df = pd.concat([balanced_df, label_subset])

    balanced_df.reset_index(drop=True)
    return balanced_df