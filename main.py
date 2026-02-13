from Decoder import Decoder, DecoderBlock
from Encoder import Encoder, EncoderBlock
from FeedForwardBlock import FeedForwardBlock
from MultiHeadAttention import MultiHeadAttentionBlock
import PositionalEncoding
import ProjectionLayer
import Transformer
import inputEmbeddings
import torch 
import torch.nn as nn
import math 


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int = 6, h: int = 8, dropout:float=0.1, d_ff: int=2048) -> Transformer:
    # Create Embeddings layers 
    src_embed = inputEmbeddings(d_model, src_vocab_size)
    tgt_embed = inputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional Embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

     # Create the decoder blocks 
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size )

    # Create the transformer 
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initialize the parameters 
    for p in transformer.paramerters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer