# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import torch
import torch.nn.functional as F 
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch_geometric.data import Data
from transformers import RobertaTokenizer

class HGDataset(Dataset):
    def __init__(self, args, text_json, fold_name):
        super(HGDataset).__init__()
        self.args = args
        self.root = os.path.join('../data', args.dataset, fold_name)
        self.file_list = os.listdir(self.root)
        self.text_json = text_json
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_file = self.file_list[idx]
        line_idx = int(data_file.split('.')[0])

        text_data = json.loads(self.text_json[line_idx])
        text = text_data['article']  
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        data = np.load(os.path.join(self.root, data_file), allow_pickle=True)
        sentence_emb = data['sentence_emb']
        entities_emb = data['entities_emb']
        num_nodes = data['num_nodes'].item()
        if len(entities_emb) == 0:
            entities_emb = np.zeros((2, 300))
            num_nodes= 2
        textG_A = data['textG_A']
        wikiG_A = data['wikiG_A']
        num_sentences = data['num_sentences'].item()
        sen_en_A_row = data['sen_en_A_row']
        sen_en_A_col = data['sen_en_A_col']
        sen_en_A_data = data['sen_en_A_data']
        sen_en_edges_num = len(sen_en_A_data)
        label = data['label'] 

        sentence_entity_edge = csr_matrix((sen_en_A_data, (sen_en_A_row, sen_en_A_col)), shape=(num_sentences, num_nodes)).toarray()

        file_id = str(data_file.split('.')[0])
        
        sen_pad_num = self.args.sen_max_len-num_sentences
        node_pad_num = self.args.en_max_len-num_nodes

        return Data(
            file_id = file_id,
            text_input_ids = text_input["input_ids"].flatten().clone().detach().type(torch.LongTensor),
            attention_mask = text_input["attention_mask"].flatten().clone().detach().type(torch.LongTensor),
            sentence_emb = F.pad(torch.FloatTensor(sentence_emb).squeeze(1), (0, 0, 0, sen_pad_num)),  #need padding
            x = F.pad(torch.FloatTensor(entities_emb), (0, 0, 0, node_pad_num)),  #need padding
            nodes_num = torch.LongTensor([num_nodes]),
            edge_index = torch.LongTensor(textG_A),
            edge_index_wiki = torch.LongTensor(wikiG_A),
            sentences_num = torch.LongTensor([num_sentences]),
            sentence_entity_edge = F.pad(torch.FloatTensor(sentence_entity_edge), (0, node_pad_num, 0, sen_pad_num)),
            y = torch.LongTensor([int(label)])
        )
    