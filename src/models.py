"""
Models.
"""

import numpy as np
import torch
import mamba
from torch import nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model


class BERT4Rec(nn.Module):

    def __init__(self, vocab_size, bert_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=bert_config['hidden_size'],
                                        padding_idx=padding_idx)
        self.transformer_model = BertModel(BertConfig(**bert_config))

        if self.add_head:
            self.head = nn.Linear(bert_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs


class SASRec(nn.Module):
    """Adaptation of code from
    https://github.com/pmixer/SASRec.pytorch.
    """

    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1,
                 num_heads=1, dropout_rate=0.1, initializer_range=0.02,
                 add_head=True):

        super(SASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(hidden_units,
                                                   num_heads,
                                                   dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # parameter attention mask added for compatibility with Lightning module, not used
    def forward(self, input_ids, attention_mask):

        seqs = self.item_emb(input_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(input_ids.shape[1])), [input_ids.shape[0], 1])
        # need to be on the same device
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))

        return outputs


class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class RNN(nn.Module):

    def __init__(self, vocab_size, rnn_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.rnn_config = rnn_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=rnn_config['input_size'],
                                        padding_idx=padding_idx)
        self.rnn = nn.GRU(batch_first=True, bidirectional=False, **rnn_config)

        if self.add_head:
            self.head = nn.Linear(rnn_config['hidden_size'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    # parameter attention mask added for compatibility with Lightning module, not used
    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        outputs, _ = self.rnn(embeds)

        if self.add_head:
            outputs = self.head(outputs)

        return outputs


class GPT4Rec(nn.Module):

    def __init__(self, vocab_size, gpt_config, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.gpt_config = gpt_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=gpt_config['n_embd'],
                                        padding_idx=padding_idx)
        self.transformer_model = GPT2Model(GPT2Config(**gpt_config))

        if self.add_head:
            self.head = nn.Linear(gpt_config['n_embd'], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs

class MambaRec(nn.Module):
    def __init__(self, item_num, maxlen=128, hidden_units=64, num_blocks=1, 
                 d_state=32, d_conv=4, expand=2, dropout_rate=0.1, 
                 initializer_range=0.02, add_head=True):
        super(MambaRec, self).__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head

        # Embedding layers for items and positions (padding_idx=0 for no-item)
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        # Initialize one or more Mamba SSM blocks
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden_units, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_blocks)
        ])

        # (Optional) initialize weights similarly to SASRec
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with normal distribution, zero-out padding embeddings."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids, attention_mask=None):
        # Input: input_ids [B, T], where 0 indicates padding.
        # Get item embeddings and add positional embeddings
        seqs = self.item_emb(input_ids)                          # [B, T, D]
        seqs *= self.item_emb.embedding_dim ** 0.5               # scale by sqrt(D)
        # Create position indices 0..T-1 for each sequence in the batch
        positions = torch.arange(seqs.size(1), device=seqs.device).unsqueeze(0).expand_as(input_ids)
        seqs += self.pos_emb(positions)                          # add positional encoding
        seqs = self.emb_dropout(seqs)
        # Mask out padded positions to zero (so they don't contribute)
        pad_mask = (input_ids == 0)                              # [B, T] bool
        if pad_mask.any():
            seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Pass through Mamba SSM layers (selective state-space modeling)
        for mamba_block in self.mamba_layers:
            seqs = mamba_block(seqs)  # each block returns [B, T, D] same shape&#8203;:contentReference[oaicite:4]{index=4}

        outputs = seqs  # [B, T, D] hidden states
        if self.add_head:
            # Project hidden states to item scores using item embedding matrix (tying weights)
            outputs = torch.matmul(outputs, self.item_emb.weight.T)  # [B, T, item_num+1]
        return outputs


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss: -log σ(pos_score - neg_score)"""
    def __init__(self):
        super().__init__()

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        diff = pos_score - neg_score
        return -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, inner_size: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        # residual + norm
        return self.norm(h + x)


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_residual: bool = True
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model, inner_size=d_model * 4, dropout=dropout)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mamba(x)
        if self.use_residual:
            h = self.norm(self.dropout(h) + x)
        else:
            h = self.norm(self.dropout(h))
        return self.ffn(h)
