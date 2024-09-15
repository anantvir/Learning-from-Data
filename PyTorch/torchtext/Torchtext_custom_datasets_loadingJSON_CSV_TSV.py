"""
Author :  Anantvir Singh
References: 

1. https://pytorch.org/text/stable/index.html
2. https://www.youtube.com/c/AladdinPersson

"""
"""
Steps

1. Specify how preprocessing should be done -> Fields
2. Use Dataset to load the data -> TabularDataset(CSV/TSV/JSON)
3. Construct iterator to do batching and padding -> BucketIterator

"""
from torchtext.data import Field, TabularDataset, BucketIterator


quote = Field(sequential = True, use_vocab = True)
# Ignore the name field in train.json because its not relevent to feed into model





















