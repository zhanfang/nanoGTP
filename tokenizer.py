from datasets import load_dataset
from pprint import pprint
from transformers import AutoTokenizer

raw_datasets = load_dataset("code_search_net", "python")

def get_training_corpus():
    dataset = raw_datasets['train']
    for i in range(0, len(dataset), 1000):
        samples = dataset[i : i + 1000]
        yield samples["whole_func_string"]


old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=52000)



example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
new_tokens = tokenizer.tokenize(example)


print(tokens)
print(new_tokens)

tokenizer.save_pretrained("code-search-net-tokenizer")
tokenizer.push_to_hub("code-search-net-tokenizer")




# pprint(raw_datasets['train'], width=80, sort_dicts=False)