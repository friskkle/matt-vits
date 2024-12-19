import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original file
input_file = "./en/validated.tsv"

df = pd.read_csv(input_file, delimiter='\t')
df['path'] = df['path'].apply(lambda x: "{}{}".format('/en/clips/', x))

# Select only the relevant columns for TTS preprocessing
filtered_df = df[['client_id', 'path', 'sentence']]

# Filter out client_id groups with fewer than 2 samples
client_counts = filtered_df['client_id'].value_counts()
sufficient_samples_df = filtered_df[filtered_df['client_id'].isin(client_counts[client_counts >= 2].index)]

# Split the data into train, validation, and test sets, stratify is to make sure each set includes a sample of every client
train_val_df, test_df = train_test_split(sufficient_samples_df, test_size=0.1, random_state=42, stratify=sufficient_samples_df['client_id'])
train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['client_id'])

# Save the splits to separate files
train_df.to_csv("./en/train_metadata.tsv", index=False, sep='|', header=False)
val_df.to_csv("./en/val_metadata.tsv", index=False, sep='|', header=False)
test_df.to_csv("./en/test_metadata.tsv", index=False, sep='|', header=False)

print("Filtered metadata saved!")
