import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np

""" 
Author : Anantvir Singh

 References :
 
 1. Stanford CS224n : https://web.stanford.edu/class/cs224n/readings/cs224n-self-attention-transformers-2023_draft.pdf
 
 2. Attention is all you need paper : https://arxiv.org/abs/1706.03762
 
 3. Aladdin Persson : https://www.youtube.com/watch?v=U0s0f995w14 : Awesome job by Aladdin. This video assumes you are fluent in PyTorch and have in depth 
 understanding of Transfomer architecture. So I have added comments along the way for someone who is new to Transformers and PyTorch
 
"""

""" 
    Lets take an example to make is easy to understand
    Assume batch_size = 32, sequence_length = 10, embedding_dimension = 512, num_heads = 8
"""
class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dimension = embedding_size // num_heads # d/k
        
        assert (self.head_dimension * num_heads == embedding_size), "Embedding size needs to be divisible by number of heads"
        
        # Values matrix V (d*d). We will split into (d*(d/k)) later
        self.V  = nn.Linear(self.embedding_size, self.embedding_size, bias = False)
        
        # Keys matrix K (d*d). We will split into (d*(d/k)) later
        self.K = nn.Linear(self.embedding_size, self.embedding_size, bias = False)
        
        # Queries matrix Q (d*d). We will split into (d*(d/k)) later
        self.Q = nn.Linear(self.embedding_size, self.embedding_size, bias = False)
        
        # Output matrix O (d*d). We will split into (d*(d/k)) later
        self.fc_out = nn.Linear(self.embedding_size, self.embedding_size)
        
    # Inputs to forward function i.e queries, keys, values have the shape (batch_size, sequence_length, embedding_dimension) 
    def forward(self, values, keys, queries, mask):
        # Input query has shape (batch_size, sequence_length, embedding_dimension)
        batch_size = queries.shape[0]
        
        # Input query has shape (batch_size, sequence_length, embedding_dimension). values and keys have same dimensions as query
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # These just represent sequence_length, all same in case of encoders 
        
        # Multiply input sequence x by Q,K,V for example xQ = (32*10*512) i.e (batch_size * sequence_length * embedding_size)
        values = self.V(values) # (batch_size * sequence_length * embedding_size)
        keys = self.K(keys) # (batch_size * sequence_length * embedding_size)
        queries = self.Q(queries) # (batch_size * sequence_length * embedding_size)
        
        # Split input embedding dimension(512 as an example) into k different heads(8 heads as an example). each dimension will be d/k i.e 512/8 = 64 in this case
        values = values.reshape(batch_size, value_len, self.num_heads, self.head_dimension)
        keys = keys.reshape(batch_size, key_len, self.num_heads, self.head_dimension)
        queries = queries.reshape(batch_size, query_len, self.num_heads, self.head_dimension)
        
        """ queries shape : (batch_size, query_len, num_heads, head dimension)
        # keys shape : (batch_size, key_len, num_heads, head dimension)
        # alpha shape we want : (batch_size, num_heads, query_len, key_len)
        
        We can reshape K,Q,V to (batch_size, num_heads, sequence_length, head_dimension) just like suggested in Stanford notes. The reason we dont do that here explicitly is because we are using einsum() and not bmm()
        so we specify this reshaping in our input and output query string in einsum() function
        # Calculate key query affinities i.e alpha scalars (xQ).(xK)^T
        # Refer to Stanford notes link above, page 10, 11 (Sequence Tensor form) i.e transpose Q,K,V from (n, k, d/k) to (k, n, d/k) for easier parallel computation of alphas for all heads """
        alpha = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])  # Instead of flattening and using torch.bmm(), einsum() is much easier
        
        if mask is not None:
            alpha = alpha.masked_fill(mask == 0, float("-1e20")) # put "-1e20" where mask == False
        
        attention = torch.softmax(alpha / (self.embedding_size ** (1/2)), dim = 3) # 3rd dimension here is key_len in alpha matrix i.e source sentence (alpha shape we want : (batch_size, heads, query_len, key_len))
        
        # Attention shape : (batch_size, num_heads, query_len, key_len)
        # value shape : (batch_size, value_len, num_heads, head dimension)
        # expected output shape : (batch_size, query_len, num_heads, head dimension) Refer to Stanford notes link above, page 10, 11 (Sequence Tensor form) i.e transpose Q,K,V from (n, k, d/k) to (k, n, d/k) for easier parallel computation of alphas for all heads
        out = torch.einsum("bhqk,bvhd->bqhd", [attention, values]).reshape(
            batch_size, query_len, self.num_heads * self.head_dimension
        ) # Here k = key length which is equal to value length in case of transformer encoder. So final output shape would be (batch_size, query_length, embedding_dimension)
        
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, forward_expansion):
        
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embedding_size, num_heads)
        
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size), # in paper forward expansion = 4
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    """
    Input shape for values, keys, queries x_1:n = (32, 10, 512). This input needs to be multiplied by K, Q, V and reshaped into k heads during self attention. 
    """    
    def forward(self, values, keys, queries, mask):
        
        attention = self.attention(values, keys, queries, mask)  # Input = (32, 10, 512), Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        skip_connection1 = attention + queries  
        """ Here we add queries and not values or keys because check the transformer diagram carefully. We need to feed iin query from masked multi-head attention.
        #they have same dimensions. Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension) """
        
        normalized_layer_output1 = self.layer_norm1(skip_connection1) # Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        dropout1_applied = self.dropout(normalized_layer_output1) # Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        feed_forward = self.feed_forward(dropout1_applied) # Input : (32, 10, 512) -> (forward_expansion) : (32, 10, 2048) -> Final output : (32, 10, 512)
        
        skip_connection2 = feed_forward + dropout1_applied # Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        normalized_layer_output2 = self.layer_norm2(skip_connection2) # Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        dropout2_applied = self.dropout(normalized_layer_output2) # Output = (32, 10, 512) -> (batch_size, sequence_length, embedding_dimension)
        
        return dropout2_applied    
    
class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_size,
                 num_layers,
                 num_heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_sentence_length
                ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_sentence_length, embedding_size)
        
        # We use ModuleList so that PyTorch is aware of this list of Transformer blocks. See this : https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    dropout = dropout,
                    forward_expansion = forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """x shape : (32,10) or (batch_size, sequence_length)"""
        batch_size, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length).to(self.device) # We need to convert position tensor to integers to perform lookup in self.position_embedding matrix declared above
        
        """ 
        Output shape of self.word_embedding(x) = (32,10,512) or (batch_size, seq_len, embedding_dimension) 
        Output shape of self.position_embedding(positions) = (32,10,512) or (batch_size, seq_len, embedding_dimension)
        """
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )
        for transformer_block in self.transformer_blocks:
            out = transformer_block(out, out, out, mask) # Since K,Q and V are all same so same parameters are input to transformer_block
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.attention = SelfAttention(embedding_size, num_heads = num_heads)
        self.transformer_block = TransformerBlock(embedding_size, num_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        
    """
    Input params
    x = Input sentence batched. Shape : (32,10) or (batch_size, sequence_length)
    value = output value matrix from encoder
    key = output key matrix from encoder
    source_mask = This is a (batch size, source sentence length) tensor that is 1 when the source sentence token is not a padding token, and 0 when it is a padding token. 
    For example, if the source sentence is: [“hello”, “how”, “are”, “you”, “?”, , ], then the mask would be [1, 1, 1, 1, 1, 0, 0]. This is to prevent unnecessary attention computation
    target_mask = boolean variable to tell our decoder to mask the attention scores
    """    
    def forward(self, x, value_from_encoder, key_from_encoder, source_mask, target_mask):
        # Step 1 : Calculate attention from input batch x
        attention = self.attention(x, x, x, target_mask)
        
        # Step 2 : Add skip connection
        new_query_from_decoder = self.dropout(self.norm(attention + x))
        
        # Step 3 : Feed the input to a transformer block created above
        out = self.transformer_block(value_from_encoder, key_from_encoder, new_query_from_decoder, source_mask)
        return out
    

class Decoder(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout,
        device,
        max_sentence_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.positional_embedding = nn.Embedding(max_sentence_length, embedding_size)
        
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embedding_size, num_heads, forward_expansion, dropout, device) for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, source_mask, target_mask):

        batch_size, sequence_length = x.shape
        positions = torch.arange(0, sequence_length).expand(batch_size, sequence_length).to(self.device)
        x = self.dropout(
            (self.word_embedding(x) + self.positional_embedding(positions))
        )
        
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, encoder_output, source_mask, target_mask)
            
        out = self.fc_out(x)
        
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocabulary_size,
        trg_vocabulary_size,
        src_pad_idx,
        trg_pad_idx,
        embedding_size = 512,
        num_layers = 6,
        num_heads = 8,
        device = "cpu",
        forward_expansion = 4,
        dropout = 0,
        max_sentence_length = 100
    ):
        super(Transformer, self).__init__()
        
        # Create Encoder Stack
        self.encoder = Encoder(
            src_vocabulary_size,
            embedding_size,
            num_layers,
            num_heads,
            device,
            forward_expansion,
            dropout,
            max_sentence_length
        )
        
        # Create Decoder stack
        self.decoder = Decoder(
            trg_vocabulary_size,
            embedding_size,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
            device,
            max_sentence_length
        )
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    # src shape : (batch_size, sequence_length)
    def make_src_mask(self, src):
        """ 
         Assume sentence = ["programming","in","pytorch","is","interesting","."]
         max_sentence_length = 10, then above sentence needs to be padded with src_pad_idx which lets say is 0. So the above sentence becomes
         ["programming","in","pytorch","is","interesting",".","<padding>","<padding>","<padding>","<padding>"] where index of "<padding>" = 0
         
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
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    
    trg = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0], 
        [1, 5, 6, 2, 4, 7, 6, 2]
        ]).to(device)
    print("Input Tensor Shape :", x.shape)
    print("Target Tensor Shape :", trg.shape)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print("Model output for a single forward pass :",out.shape)