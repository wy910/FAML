# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import math
from torch_scatter import scatter_mean, scatter_max
from transformers import RobertaModel
import torch.nn.init as init

#entity-level
class text_wiki_GCN(nn.Module):
    def __init__(self, args):
        super(text_wiki_GCN, self).__init__()
        self.args = args

        self.textconv1 = GCNConv(args.entity_dim, args.hid_dim)
        self.textconv2 = GCNConv(args.hid_dim, args.out_dim)

        self.wikiconv1 = GCNConv(args.entity_dim, args.hid_dim)
        self.wikiconv2 = GCNConv(args.hid_dim, args.out_dim)

        self.w = nn.Linear(args.out_dim, args.out_dim)

    def forward(self, data):
        #id = data.file_id
        x = data.x 
        edge_index = data.edge_index
        edge_index_wiki = data.edge_index_wiki

        text_out = F.gelu(self.textconv1(x, edge_index))
        #text_out = F.dropout(text_out, p = 0.2, training = self.training)
        text_out = F.gelu(self.textconv2(text_out, edge_index))

        wiki_out = F.gelu(self.wikiconv1(x, edge_index_wiki))
        #wiki_out = F.dropout(wiki_out, p = 0.2, training = self.training)
        wiki_out = F.gelu(self.wikiconv2(wiki_out, edge_index_wiki))
        
        diff_out = self.w(text_out-wiki_out)#.view(self.args.batch_size, -1, self.args.out_dim)
        diff_out, _ = scatter_max(diff_out, data.batch, dim=0)
        #print('diff_out GCN', diff_out)
        #diff_out = F.max_pool2d(diff_out.unsqueeze(1), kernel_size=(self.args.en_max_len, 1)).squeeze(1).squeeze(1) #[batch_size, out_dim]

        return text_out, diff_out
    
# sentence-level
class PositionalEncoding(nn.Module):
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        self.args = args

        self.dropout = nn.Dropout(p = args.dropout)
        pe = torch.zeros(args.sen_max_len, args.out_dim)
        position = torch.arange(0, args.sen_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.out_dim, 2).float() * (-math.log(10000.0) / args.out_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe:[max_len, out_dim]
        pe = pe.unsqueeze(0).transpose(0, 1) #pe:[max_len, 1, out_dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x:[batch_size, max_sen_len, out_dim]
        #x = x.view(self.batch_size, self.sen_max_len, self.out_dim) 
        x = x.transpose(0, 1)  #x:[sem_max_len, batch_size, out_dim]
        x = x + self.pe[:x.size(0), :]
        x = x.transpose(0, 1)
        return self.dropout(x)  #x:[batch_size, max_sen_len, out_dim]
    

class MultiHeadAttention(nn.Module): 
    def __init__(self, args, mean_value, var_value):
        super(MultiHeadAttention, self).__init__() 
        assert args.out_dim % args.heads == 0 # We assume d_v always equals d_k 
        self.args = args
        self.d_k = args.out_dim // args.heads    

        self.W_Q = nn.Linear(args.out_dim, self.d_k * args.heads)
        self.W_K = nn.Linear(args.out_dim, self.d_k * args.heads)
        self.W_V = nn.Linear(args.out_dim, self.d_k * args.heads)
        self.linear = nn.Linear(args.heads * self.d_k, args.out_dim)

        self.layer_norm = nn.LayerNorm(args.out_dim) 

        self.attn = None 
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, Q, K, V, attn_mask=None):
        #Q: [batch_size, len_q, d_model], K: [batch_size, len_k, d_model], V: [batch_size, len_k, d_model]
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.args.heads, self.d_k).transpose(1,2)  # q_s: [batch_size, n_heads, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.args.heads, self.d_k).transpose(1,2)  # k_s: [batch_size, n_heads, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.args.heads, self.d_k).transpose(1,2)  # v_s: [batch_size, n_heads, len_k, d_v]
 
        #input attn_mask batch_size, len_q, len_k，output attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.heads, 1, 1)
        
        #context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.args.heads * self.d_k) # context: [batch_size, len_q, n_heads * d_v]
        output = self.linear(context)
        output = self.dropout(output)
        return self.layer_norm(output + residual), attn # output: [batch_size, len_q, d_model]
    

    def ScaledDotProductAttention(self, query, key, value, mask = None, dropout = None):
        #query, key, value: [batch_size, head, seq_len, dim]
        #d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k) 
        if mask is not None: 
            scores = scores.masked_fill(mask == 1, -1e9) 
        p_attn = F.softmax(scores, dim = -1) 
        
        if dropout is not None: 
            p_attn = dropout(p_attn) 
        return torch.matmul(p_attn, value), p_attn
    

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args, mean_value, var_value):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(in_channels=args.out_dim, out_channels=args.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.out_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(args.out_dim)
 
    def forward(self, x):
        residual = x # inputs : [batch_size, len_q, d_model]
        output = F.relu((self.conv1(x.transpose(1, 2))))
        output = self.conv2(output).transpose(1, 2)

        return self.layer_norm(output + residual)



class EncoderLayer(nn.Module):
    def __init__(self, args, mean_value, var_value):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.enc_self_attn = MultiHeadAttention(args, mean_value, var_value)
        self.pos_ffn = PoswiseFeedForwardNet(args, mean_value, var_value)
 
    def forward(self, x, attn_mask):
        ## enc_inputs: [batch_size, seq_len_q, d_model] 
        enc_outputs, attn = self.enc_self_attn(x, x, x, attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, len_q, d_model]
        return enc_outputs, attn
    

class sen_Encoder(nn.Module):
    def __init__(self, args, mean_value, var_value):
        super(sen_Encoder, self).__init__()
        self.args = args
        self.dropout = args.dropout

        self.pos_emb = PositionalEncoding(args)
        self.layers = nn.ModuleList([EncoderLayer(args, mean_value, var_value) for _ in range(args.n_layers)]) 
 
    def forward(self, x):
        ## enc_inputs： [batch_size*source_len, d_model]
        x = x.view(self.args.batch_size, self.args.sen_max_len, -1) #[batch_size, q_len, d_model]
        #print('sen_encoder_inner x', x.shape)

        enc_outputs = self.pos_emb(x)  #[batch_size, q_len, d_model]
        #print('sen_encoder_inner position x', enc_outputs.shape)
 
        enc_self_attn_mask = self.get_attn_pad_mask(x, x)   # [batch_size, len_q, len_k]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
    def get_attn_pad_mask(self, query, key):
        #query/key:[batch_size, seq_len, d_model]
        query = torch.sum(query, dim=-1)  #[batch_size, seq_len]
        key = torch.sum(key, dim=-1)
        batch_size, len_q = query.size()
        batch_size, len_k = key.size()
        # eq(zero) is PAD token
        pad_attn_mask = key.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], one is masking 
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class HGmodel(nn.Module):
    def __init__(self, args):
        super(HGmodel, self).__init__()
        self.args = args

        self.roberta = RobertaModel.from_pretrained(args.model_path)
        last_layer_params = self.roberta.pooler.dense.weight.data
        self.mean_value = torch.mean(last_layer_params)
        self.variance_value = torch.var(last_layer_params)

        self.text_fc = nn.Linear(args.text_in_dim, args.out_dim)

        self.text_wiki_GCN = text_wiki_GCN(args)

        self.sen_fc = nn.Linear(args.sen_in_dim, args.out_dim)
        self.sen_gate_w = nn.Linear(2*args.out_dim, args.out_dim)
        self.senG = sen_Encoder(args, self.mean_value, self.variance_value)

        self.text_gate_w = nn.Linear(2*args.out_dim, args.out_dim)

        #self.layer_norm1 = nn.LayerNorm(args.out_dim)
        #self.layer_norm2 = nn.LayerNorm(args.out_dim)

        self.classifier_fc1 = nn.Linear(2*args.out_dim, args.out_dim)
        self.classifier_fc2 = nn.Linear(args.out_dim, 2)

        #self.init_transformer_weights()
    
    def init_transformer_weights(self):
        for name, param in self.senG.named_parameters():
            if 'weight' in name:
                if 'attn' in name:  
                    init.xavier_normal_(param)  
                else:
                    init.kaiming_normal_(param)  
            elif 'bias' in name:
                init.zeros_(param)  

    def forward(self, data):
        #print('step0 read data...')
        file_id = data.file_id
        sentence_input = data.sentence_emb
        sentence_entity_edge = data.sentence_entity_edge

        text_input_ids = data.text_input_ids.reshape(self.args.batch_size, -1)
        attention_mask = data.attention_mask.reshape(self.args.batch_size, -1)
        text_out = self.roberta(text_input_ids, attention_mask)
        text_out = text_out.pooler_output
        #print('roberta_out', text_out.shape)
        text_emb = self.text_fc(text_out)
        
        #print('step1 GCN...')
        entity_emb, diff_out = self.text_wiki_GCN(data)
        
        #print('step2 entity max_pool...')
        sen_en_emb = self.sen_en_map(sentence_entity_edge, entity_emb)
        #print('sen_en_emb', sen_en_emb)

        #print('step3 sen gate...')
        sen_emb = self.sen_fc(sentence_input)
        #print('sen_emb', sen_emb.shape)
        sen_en_cat = torch.cat((sen_en_emb, sen_emb), dim=1)
        #print('sen_en_cat',sen_en_cat.shape)
        sen_gate = F.sigmoid(self.sen_gate_w(sen_en_cat))
        #print('sen_gate', sen_gate.shape)
        sen_gated_emb = sen_gate*sen_en_emb + (1-sen_gate)*sen_emb 
        #print('sen_gated_emb', sen_gated_emb.shape)
        #sen_gated_emb = self.layer_norm1(sen_emb + sen_en_emb)

        #print('step4 sen transformer...')
        sen_out, attns = self.senG(sen_gated_emb)  #[batch_size, sen_max_len, out_dim]
        #print('sen_out', sen_out.shape)
        #sen_max_pool_out, _ = scatter_max(sen_out, data.batch, dim=0)
        sen_max_pool_out = F.max_pool2d(sen_out.unsqueeze(1), kernel_size=(self.args.sen_max_len, 1)).squeeze(1).squeeze(1) 
        #print('sen_max_pool_out', sen_max_pool_out.shape)
        
        #print('step5 text gate...')
        text_sen_cat = torch.cat((sen_max_pool_out, text_emb), dim=1)
        #print('text_sen_cat', text_sen_cat.shape)
        text_gate = F.sigmoid(self.text_gate_w(text_sen_cat))
        #print(text_gate)
        text_gated_emb = text_gate*sen_max_pool_out + (1-text_gate)*text_emb
        #print('text_gated_emb', text_gated_emb.shape)
        #text_gated_emb = self.layer_norm2(text_emb + sen_max_pool_out)

        #print('step6 text cat diff..')
        out = torch.cat((text_gated_emb, diff_out), dim=1)
        #print('out', out.shape)
        out = F.relu(self.classifier_fc1(out))
        out = self.classifier_fc2(out)
        pred = F.softmax(out, dim=1)
        return out, pred
    
    def sen_en_map(self, sentence_entity_edge, entity_emb):
        # sentence_entity_edge:[batch_size*sen_len, entity_len]
        # entity_emb:[batch_size*entity_len, entity_dim]
        entity_dim = entity_emb.size(-1)
        sentence_entity_edge = sentence_entity_edge.view(self.args.batch_size, self.args.sen_max_len, -1)
        entity_emb = entity_emb.view(self.args.batch_size, self.args.en_max_len, -1)
        sentence_entity_num = torch.sum(sentence_entity_edge, dim=-1, keepdim=True)
        e = torch.full((self.args.batch_size, self.args.sen_max_len, 1), 0.0001, dtype=torch.float).to(sentence_entity_edge.device)#.cuda(device=sentence_entity_edge.device)
        sentence_entity_num += e
        sen_en = torch.matmul(sentence_entity_edge, entity_emb)
        sen_en /= sentence_entity_num
        return sen_en.view(-1, entity_dim)
       

