"""
Author :  Anantvir Singh
References: 

1. https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
2. https://pytorch.org/tutorials/beginner/translation_transformer.html

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

# Place-holders
token_transform = {}        # Holds tokenizer function from spacy. To call tokenizer for every sentence call token_transform["en"](english sentence)
vocab_transform = {}        # Holds vocabulary for en and de

# ------------------------------------ Step 1 : Get first tranformation i.e tokenizer ------------------------------------
token_transform["en"] = get_tokenizer("spacy", language="de_core_news_sm")
token_transform["de"] = get_tokenizer("spacy", language="en_core_web_sm")

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {"en": 0, "de": 1}
    # data_sample[0] = english sentence -> Two young, White males are outside near many bushes.
    # data_sample[1] = german sentence -> "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# ------------------------------------ Step 2 : Get second tranformation i.e build vocab ------------------------------------
for language in ["en","de"]:
    # get training data Iterator
    train_iter = Multi30k(split = "train", language_pair=("en", "de"))

    # Create torchtext's Vocab object
    vocab_transform[language] = build_vocab_from_iterator(yield_tokens(train_iter, language),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for language in ["en", "de"]:
  vocab_transform[language].set_default_index(UNK_IDX)

# ------------------------------------ Step 3 : Get third tranformation i.e add SOS, EOS to a sentence and create tensor ------------------------------------

# Function to add SOS/EOS to input list of token ids and create tensor from that list
def transform_to_tensor(token_ids : List[int]):
    return torch.cat(
        (torch.tensor([SOS_IDX]),
         torch.tensor(token_ids),
         torch.tensor([EOS_IDX])
         ))

# ------------------------------------ Step 4 : Apply above 3 transforms to the input list of integers (tensor) ------------------------------------

# Helper function to club together sequential transforms
def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input
    return func

text_transform = {}
for language in ["en","de"]:
    text_transform[language] = sequential_transforms(token_transform[language], # Tokenization
                                                     vocab_transform[language], # Create Vocab
                                                     transform_to_tensor             # Add SOS/EOS and create tensor
                                                     )

# ------------------------------------ Step 5 : Colate data and create batches using collate_fun()  ------------------------------------
# Function to collate individual sentences in batch tensors
def collate_fn(batch):
    source_batch, target_batch = [], []
    for source_sample, target_sample in batch:
        source_batch.append(text_transform["en"](source_sample.rstrip("\n")))  # Remove trailing \n
        target_batch.append(text_transform["de"](target_sample.rstrip("\n")))

    source_batch = pad_sequence(source_batch, padding_value = PAD_IDX)
    target_batch = pad_sequence(target_batch, padding_value = PAD_IDX)
    return source_batch, target_batch

train_iter = Multi30k(split='train', language_pair=("en", "de"))
train_dataloader = DataLoader(train_iter, batch_size=10, collate_fn=collate_fn)

for source,target in train_dataloader:
    print(source)
    print(target)


class Transformer(nn.Module):
    def __init__(
            self,
            embedding_dimension,
            source_vocab_size,
            target_vocab_size,
            source_pad_idx,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_sentence_length,
            device
        ):
        super(Transformer, self).__init__()

        # Source word embedding and source positional embedding
        self.source_word_embedding = nn.Embedding(source_vocab_size, embedding_dimension) # Shape : (Source_vocab_size X embedding_dimension )
        self.source_position_embedding = nn.Embedding(max_sentence_length, embedding_dimension) # Shape : (max_sentence_length X embedding_dimension )

        # Target word embedding and source positional embedding
        self.target_word_embedding = nn.Embedding(target_vocab_size, embedding_dimension) # Shape : (target_vocab_size X embedding_dimension )
        self.target_position_embedding = nn.Embedding(max_sentence_length, embedding_dimension) # Shape : (max_sentence_length X embedding_dimension )

        self.device = device
        
        self.transformer = nn.Transformer(
            d_model = embedding_dimension,
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward = forward_expansion,
            dropout = dropout
        )

        self.fc_out = nn.Linear(embedding_dimension, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.source_pad_idx = source_pad_idx

    def make_source_mask(self, source_text):
        # Input source_text shape : (src_len, batch_size) but nn.Transformer takes input of (batch_size, src_len)
        source_mask = source_text.transpose(0,1) == self.source_pad_idx
        return source_mask
    
    def forward(self, source, target):
        pass



























