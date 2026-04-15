from datasets import load_dataset
import os

"""
reference: https://medium.com/@patelneha1495/recommendation-system-in-python-using-als-algorithm-and-apache-spark-27aca08eaab3
"""


# 1. load review data and save as parquet 
parquet_name = 'my_amazon_books_sample.parquet'
if not os.path.exists(parquet_name):
    print(f'The review dataset is being generated------')
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Books", trust_remote_code=True)
    cols_to_keep = ["user_id", "parent_asin", "rating", "text"]
    small_ds = dataset["full"].select_columns(cols_to_keep)
    sampled_ds = small_ds.select(range(int(len(small_ds) * 0.10)))
    sampled_ds.to_parquet(f'{parquet_name}')
else:
    print(f'{parquet_name} has been there----------')


# 2. load meta data and save as parquet
parquet_name = 'my_amazon_books_meta_sample.parquet'
if not os.path.exists(parquet_name):
    print(f'The meta dataset is being generated------')
    meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Books", trust_remote_code = True)
    cols_to_keep = ["title", "parent_asin", "subtitle", "description", "author"]
    sampled_ds = meta_dataset["full"].select_columns(cols_to_keep)
    sampled_ds.to_parquet('my_amazon_books_meta_sample.parquet')
else:
    print(f'{parquet_name} has been there----------')