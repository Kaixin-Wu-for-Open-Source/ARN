import itertools
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

class RefineAttention_Merge(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        self.in_proj_weight_q, self.in_proj_weight_k, self.in_proj_weight_v = None, None, None

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
            self.in_proj_bias_q, self.in_proj_bias_k, self.in_proj_bias_v = None, None, None
        else:
            self.register_parameter('in_proj_bias', None)

        self.scaling = embed_dim**-0.5
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)

    def in_proj_q(self, query):
        if self.in_proj_weight_q is None:
            self.in_proj_weight_q = self.in_proj_weight[:self.embed_dim, :]
            self.in_proj_bias_q = self.in_proj_bias[:self.embed_dim]
        return F.linear(query, self.in_proj_weight_q, self.in_proj_bias_q)

    def in_proj_k(self, key):
        if self.in_proj_weight_k is None:
            self.in_proj_weight_k = self.in_proj_weight[self.embed_dim:2*self.embed_dim, :]
            self.in_proj_bias_k = self.in_proj_bias[self.embed_dim:2*self.embed_dim]
        return F.linear(key, self.in_proj_weight_k, self.in_proj_bias_k)

    def in_proj_v(self, value):
        if self.in_proj_weight_v is None:
            self.in_proj_weight_v = self.in_proj_weight[2*self.embed_dim:, :]
            self.in_proj_bias_v = self.in_proj_bias[2*self.embed_dim:]
        return F.linear(value, self.in_proj_weight_v, self.in_proj_bias_v)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def forward(self, x, cross_attn_memory, cross_attn_weights, enc_outs, slf_attn_memory, slf_attn_weights, \
            enc_slf_attn_weights, enc_slf_attn_memory, incremental_state=None):
        src_len = enc_outs.size(0)
        tgt_len, bsz, embed_dim = x.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)  
        else:
            saved_state = None

        if saved_state is not None:
            if 'prev_value' in saved_state:
                slf_linear = self.in_proj_q(x)
                x_tmp = torch.cat((saved_state['prev_value'], slf_linear), dim=0)    
            else:
                enc_outs_linear = self.in_proj_k(enc_outs)
                slf_linear = self.in_proj_q(x)
                x_tmp = torch.cat((enc_outs_linear, slf_linear), dim=0)              
            saved_state['prev_value'] = x_tmp
            self._set_input_buffer(incremental_state, saved_state)
        else:
            enc_outs_linear = self.in_proj_k(enc_outs)
            slf_linear = self.in_proj_q(x)
            x_tmp = torch.cat((enc_outs_linear, slf_linear), dim=0)                  

        x_tmp_format = x_tmp.contiguous().transpose(0, 1).view(bsz, -1, embed_dim)
        cur_enc_slf_attn_memory = torch.bmm(enc_slf_attn_weights, x_tmp_format)                         
        cur_enc_slf_attn_memory = cur_enc_slf_attn_memory.transpose(0, 1)                           

        alpha = torch.relu(self.scaling * self.in_proj_v(torch.max(cur_enc_slf_attn_memory, enc_slf_attn_memory)))
        new_enc_slf_attn_memory = cur_enc_slf_attn_memory + alpha * enc_slf_attn_memory
        new_enc_slf_attn_memory = F.dropout(new_enc_slf_attn_memory, p=self.dropout, training=self.training)
        output = slf_linear + new_enc_slf_attn_memory

        return output, new_enc_slf_attn_memory

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
