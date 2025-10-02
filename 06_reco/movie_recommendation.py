import sys
import logging
import numpy as np
import pandas as pd

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.models.sar import SAR
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

print(f"System version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = "100k"

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE
)

# Convert the float precision to 32-bit in order to reduce memory consumption
data["rating"] = data["rating"].astype(np.float32)

print("Columns after loading the data:")
print(data.head())

train, test = python_stratified_split(data, ratio=0.75, col_user="userID", col_item="itemID", seed=42)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')

model = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",
    time_decay_coefficient=30,
    timedecay_formula=True,
    normalize=True
)

with Timer() as train_time:
    model.fit(train)

print(f"Took {train_time.interval} seconds for training.")

with Timer() as test_time:
    top_k = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

print(f"Took {test_time.interval} seconds for prediction.")

print("Recommended items:")
print(top_k.head())

# Evaluate how well SAR performs
eval_map = map(test, top_k)
eval_ndcg = ndcg_at_k(test, top_k, k=TOP_K)
eval_precision = precision_at_k(test, top_k, k=TOP_K)
eval_recall = recall_at_k(test, top_k, k=TOP_K)

print(f"MAP: {eval_map}")
print(f"NDCG@k: {eval_ndcg}")
print(f"Precision@k: {eval_precision}")
print(f"Recall@k: {eval_recall}")
