"""
Author :  Anantvir Singh

References: 
1. https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
2. https://pytorch.org/tutorials/beginner/translation_transformer.html

Useful links :
1. https://datascience.stackexchange.com/questions/122077/confused-about-output-shape-of-the-transformer
2. https://datascience.stackexchange.com/questions/117949/requirements-for-variable-length-output-in-transformer/117959#117959
3. https://medium.com/@hunter-j-phillips/the-decoder-8882c33de69a
4. https://medium.com/nerd-for-tech/nlp-zero-to-one-transformers-part-13-30-5cd5a3ddd93b
5. https://betterprogramming.pub/a-guide-on-the-encoder-decoder-model-and-the-attention-mechanism-401c836e2cdb

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}        # Holds tokenizer function from spacy. To call tokenizer for every sentence call token_transform["en"](untokenized english sentence)
vocab_transform = {}        # Holds vocabulary for "en" and "de" languages

# ------------------------------------ Step 1 : Get first tranformation i.e tokenizer ------------------------------------
token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="de_core_news_sm")
token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    # data_sample[0] = english sentence -> Two young, White males are outside near many bushes.
    # data_sample[1] = german sentence -> "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    for data_sample in data_iter: # data_sample is a tuple
        yield token_transform[language](data_sample[language_index[language]])
        # This is what we yield : ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

# ------------------------------------ Step 2 : Get second tranformation i.e build vocab ------------------------------------
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # get training data Iterator
    train_iter = Multi30k(split = "train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

    # Create torchtext's Vocab object
    vocab_transform[language] = build_vocab_from_iterator(yield_tokens(train_iter, language),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
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
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[language] = sequential_transforms(token_transform[language], # Tokenization
                                                     vocab_transform[language], # Create Vocab
                                                     transform_to_tensor             # Add SOS/EOS and create tensor
                                                     )

# ------------------------------------ Step 5 : Colate data and create batches using collate_fun()  ------------------------------------
# Function to collate individual sentences in batch tensors
def collate_fn(batch):
    source_batch, target_batch = [], []
    for source_sample, target_sample in batch:
        source_batch.append(text_transform[SRC_LANGUAGE](source_sample.rstrip("\n")))  # Remove trailing \n
        target_batch.append(text_transform[TGT_LANGUAGE](target_sample.rstrip("\n")))

    source_batch = pad_sequence(source_batch, padding_value = PAD_IDX)
    target_batch = pad_sequence(target_batch, padding_value = PAD_IDX)
    return source_batch, target_batch

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
        """ 
         Assume sentence = ["programming","in","pytorch","is","interesting","."]
         max_sentence_length = 10, then above sentence needs to be padded with src_pad_idx which lets say is 0. So the above sentence becomes
         ["programming","in","pytorch","is","interesting",".","<padding>","<padding>","<padding>","<padding>"] where index of "<padding>" = 0
         source_key_mask will make sure that <padding> tokens are not attended to during self attention
        for example if 
        x = torch.tensor(
        [
            [1,0,3,4,5],
            [6,7,8,0,10],
            [11,0,13,14,15],
            [16,17,18,0,20]
        ])
        then src_mask = (x != 0) = 
        [[ True, False,  True,  True,  True],
        [ True,  True,  True, False,  True],
        [ True, False,  True,  True,  True],
        [ True,  True,  True, False,  True]]
        """
        return source_text == self.source_pad_idx

    
    def forward(self, source, target):
        """Last batch has float token indices e.g 2.0 instead of 2, so converted them to long to prevent following error I encountered: Expected tensor for argument #1 'indices' to 
        have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)"""
        source = source.long()  
        target = target.long()
        source_seq_length, source_batch_size = source.shape
        target_seq_length, target_batch_size = target.shape

        source_positions = torch.arange(0, source_seq_length).expand(source_batch_size, source_seq_length).T.to(self.device)
        target_positions = torch.arange(0, target_seq_length).expand(target_batch_size, target_seq_length).T.to(self.device)

        source_embeddings = self.dropout(
            (self.source_word_embedding(source) + self.source_position_embedding(source_positions))
        )
        #source_embeddings.to(self.device).long()
        target_embeddings = self.dropout(
            (self.target_word_embedding(target) + self.target_position_embedding(target_positions))
        )
        #target_embeddings.to(self.device).long()
        
        # tgt_tkn_embd = self.target_word_embedding(target)
        
        # tgt_pos_embd = self.target_position_embedding(target_positions)
        
        # target_embeddings = self.dropout(tgt_tkn_embd + tgt_pos_embd)
        source_key_padding_mask = self.make_source_mask(source).T # Transpose because nn.Transformer expects input shape: (N,S) or (batch_size, source_sentence_length)
        target_mask = self.transformer.generate_square_subsequent_mask(target_seq_length).to(self.device)

        out = self.transformer(
            source_embeddings,
            target_embeddings,
            src_key_padding_mask = source_key_padding_mask,
            tgt_mask = target_mask
        )
        out = self.fc_out(out)
        return out

    def encode(self, source):
        # source shape : (sequence_length, source_batch_size) for example (100,32)
        source_sequence_length, source_batch_size = source.shape

        source_positions = torch.arange(0, source_sequence_length).expand(source_batch_size, source_sequence_length).T
        source_embeddings = self.dropout(
            (self.source_word_embedding(source) + self.source_position_embedding(source_positions))
        )
        print(source_embeddings.shape)
        return self.transformer.encoder(source_embeddings)
    

# Setup training loop

load_model = False
save_model = True

# Training hyperparams
num_epochs = 5
learning_rate = 3e-4
batch_size = 128
#print(vocab_transform["de"].vocab.lookup_token(16553))
# Model hyperparams
source_vocab_size = len(vocab_transform[SRC_LANGUAGE])
target_vocab_size = len(vocab_transform[TGT_LANGUAGE])
embedding_dimension = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.10
max_length = 5000
forward_expansion = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Tensorboard

model = Transformer(
    embedding_dimension,
    source_vocab_size,
    target_vocab_size,
    PAD_IDX,
    num_heads,
    num_encoder_layers,
    num_encoder_layers,
    forward_expansion,
    dropout,
    max_length,
    device
).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


# train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
# train_dataloader = DataLoader(train_iter, batch_size=4, collate_fn=collate_fn) # Generates batches with shape : (sequence_length, batch_size)
        
# for source,target in train_dataloader:
#     print("target :",target)

#     trg = target[:-1,:]
#     print(trg)
#     print("finish one iteration")

def train_epoch(model, optimizer):

    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn) # Generates batches with shape : (sequence_length, batch_size)
        
    for source,target in train_dataloader:
        
        source = source.to(device)  # (Sequence_len, batch_size)
        target = target.to(device) # (sequence_len, batch_size). We need to remove last token which is either <eos> or <pad> since in a decoder, length of input should be equal to length of output
        target_to_input = target[:-1,:].to(device) # Remove last token, keep all sentences in batch

        # Masks are generated inside forward method of Transformer class. Can be generated here as well

        output = model(source, target_to_input)
        #model.encode(source)

        # make gradients = 0 for every batch before calculating new ones
        optimizer.zero_grad()

        """
        Assume if sentence : ["<sos>","i","am","watching","television",".","<eos>"]
        Then what we feed into decoder : ["<sos>","i","am","watching","television","."] hence target_to_input = target[:-1,:] # Remove last token
        What output we expect : ["i","am","watching","television",".","<eos>"] hence true labels = target[1:,:] # Remove first token and keep rest
        Why do we do this ? : Because length of input to decoder = length of output from decoder. Also we want decoder to learn to predict the
        next word given the words so far """
        target_output_expected = target[1:,:].type(torch.LongTensor).to(device)

        """
        Input to loss 
        predictions : (N,C) or (batch_size*seq_len, trg_vocab_size) or (16*4, 10837) or (64,10837)
        actual labels : (N) or (batch_size*seq_len) or (16*4) or (64)
        output : (N) or (batch_size*seq_len) or (16*4) or (64)
        loss_fn knows true labels (indices in vocab for target sentence) and it knows predictions and vocab size.
        it will apply softmax and calculate loss
        """
        loss = loss_fn(output.reshape(-1, output.shape[-1]), target_output_expected.reshape(-1))

        loss.backward() # compute gradients 

        optimizer.step() # Take step down the gradient

        losses += loss.item()
    
    return losses / len(list(train_dataloader))


def evaluate(self):

    # Turn models evaluation mode on
    model.eval()
    losses = 0

    validation_iterator = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)) 
    validation_dataloader = DataLoader(validation_iterator, batch_size=batch_size, collate_fn=collate_fn)

    for source, target in validation_dataloader:
        source = source.to(device)
        target = target.to(device)
        
        # Target input to send into the decoder
        target_input = target[:-1,:]

        output = model(source, target_input)

        target_output_expected = target[1:,:]

        loss = loss_fn(output.reshape(-1,output.shape[-1]), target_output_expected.reshape(-1))

        losses +- loss.item()

    return losses / len(list(validation_dataloader)) 

from timeit import default_timer as timer
NUM_EPOCHS = 10

# Training loop
for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer)
    end_time = timer()
    val_loss = evaluate(model)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
                

# def greedy_decode(model, source, source_mask, max_len, start_symbol):
#     source = source.to(device)
#     source_mask = source_mask.to(device)

    



















