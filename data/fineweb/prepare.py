"""
FineWeb-Edu dataset (sample-10BT) preparation script.
This is a high-quality dataset for pretraining LLMs, significantly better than OpenWebText.
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
# Note: set to 1 to avoid "Pickler._batch_setitems" error on Python 3.12+ with some library versions
num_proc = 8 

# number of workers in load_dataset() call
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # 1. Load the dataset
    # HuggingFaceFW/fineweb-edu is a high-quality educational dataset
    # sample-10BT is a 10 Billion Token subset, which is manageable but still large (~20GB)
    print("Loading FineWeb-Edu (sample-10BT) dataset...")
    try:
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", num_proc=num_proc_load_dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and 'datasets' installed.")
        exit(1)

    # 2. Split into train and val
    # The sample-10BT only has a 'train' split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # 3. Tokenize
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        return {'ids': ids, 'len': len(ids)}

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text', 'id', 'dump', 'url', 'date', 'file_path', 'language', 'language_score', 'token_count'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 4. Save to bin files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"saved {split}.bin with {arr_len} tokens")
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        # FineWeb is huge, so we might exceed uint16 if vocab size > 65535, but GPT-2 vocab is 50257
        # so uint16 is safe for token IDs.
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Write in batches
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
