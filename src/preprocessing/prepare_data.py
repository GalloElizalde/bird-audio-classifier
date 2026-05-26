from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import numpy as np
from utils.dictionary import DICTIONARY_LABEL2WEIGHT

# path to main dataframe 
path_df = "~/1_machine_learning_projects/BIRDS/data/train.csv" 

# Load table with data info 
df = pd.read_csv(path_df) 
print(f"The dim of train table before transforming data is {df.shape}")
print(f"With columns: {list(df.columns)}")

# PREPROCESS DATA
# Delete specific columns
columns_deleted = ['common_name', "class_name", 'scientific_name', 
                   'inat_taxon_id', 'license', 'url', "latitude","longitude", "type"] 
df = df.drop(columns=columns_deleted).copy()

# Transform "collection" to one-hot for future partition and validation, and delete collection column
mask = (df["collection"] == "iNat")
df["iNat"] =  np.array(mask).astype(int)
df["XC"] = np.array(~mask).astype(int)
df = df.drop(columns="collection").copy()

# Load taxonomy table and count all classes (234)
path_taxonomy = "~/1_machine_learning_projects/BIRDS/data/taxonomy.csv"
df_taxonomy =  pd.read_csv(path_taxonomy)
primary_label = df_taxonomy["primary_label"].unique()
print(f"The total number of classes is {len(primary_label)}")

# Add new column "sample_weight" using precomputed dictionary
df["sample_weight"] = df["primary_label"].map(DICTIONARY_LABEL2WEIGHT)

# Print Final Dataframe to use
print(f"\nThe dim of table after transforming data is {df.shape}")
print(f"With columns: {list(df.columns)}")

# S P L I T 
# fill Missing values of author with "unknown"
df["author"] = df["author"].fillna("unknown").astype(str)

# Split data acording to y = "primary_label", and grouped by "autor"
folds = 5
cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=666)
df["fold"] = -1
for fold, (_, val_idx) in enumerate(cv.split(df, y=df["primary_label"], groups=df["author"])):
    df.loc[val_idx, "fold"] = fold

# Look how many vals per fold
print("Data per fold:")
print(df["fold"].value_counts().sort_index())

# Save dfs
path_to_save = "./data/df_with_folds.csv"
df.to_csv(path_to_save, index=False)
print(f"\nThe train dataframes were created and saved succesfully at {path_to_save}")
print(f"Number of features: {len(df.columns)-1}")

