import os
from transformers import LongformerTokenizer, LongformerModel, RobertaModel, RobertaTokenizer
import torch
import json
import re
import numpy as np
import gensim
from itertools import islice
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'grover'
text_path = os.path.join('../data', dataset_name, 'train_clean.jsonl')
textG_path = os.path.join('../data', dataset_name, 'train_clean_textG.jsonl')
wikiG_path = os.path.join('../data', dataset_name, 'train_clean_wikiG.jsonl')
text2wiki_path = os.path.join('../data', dataset_name, 'train_clean_text2wiki.jsonl')


roberta_model_path = '../model/RoBERTa-large'
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
roberta_model = RobertaModel.from_pretrained(roberta_model_path)
roberta_model.to(device)

word2vec_model_path = '../model/word2vec/GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
unk = np.random.normal(0, 1, size=(300))

glove_path = '../model/glove/glove.840B.300d.txt'


def get_document_emb(text):
    input = roberta_tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = roberta_model(**input)
    pooler_output = output.pooler_output
    return pooler_output.cpu().numpy() 


def get_sentence_emb(sentences):
    sentences_emb = []
    for i, s in enumerate(sentences):
        input = roberta_tokenizer(s, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            output = roberta_model(**input)
        pooler_output = output.pooler_output
        sentences_emb.append(list(pooler_output.cpu().numpy()))
    return np.array(sentences_emb)  #shape:[sen_len, 1, 768]


def get_entity_emb_w2v(entities, unk):
    #print('entities', entities)
    entities_emb = []
    for i, entity in enumerate(entities):
        word_list = entity.split(' ')
        if len(word_list) > 1:
            e_emb = []
            for w in word_list:
                try:
                    e_emb.append(np.array(word2vec_model[w]))
                except:
                    e_emb.append(unk)
            entities_emb.append(np.mean(np.array(e_emb), axis=0))
        else:
            try:
                entities_emb.append(np.array(word2vec_model[entity]))
            except:
                entities_emb.append(unk)
    return np.array(entities_emb)   #shape:[entity_num, 300]
        
         
def wiki_norm(wikis):
    wiki_en_norm = []
    for e in wikis:
        if len(e) > 1:
            new_w_list = []
            wlist = e.split(' ')
            for i in range(len(wlist)):
                w_norm = re.sub(r"[(.*)]", "", wlist[i])
                new_w = re.sub(r"[^a-zA-Z]", "", w_norm)
                new_w_list.append(new_w)
            e_norm = '_'.join(new_w_list)
            e_norm = e_norm.strip('_')
            wiki_en_norm.append(e_norm)
        else:
            wiki_en_norm.append(e)
    return wiki_en_norm


def gen_3G(line_idx, text2wiki, entities, wikis, sentences):
    #text2wiki = json.loads(lines4[line_idx])

    num_nodes = len(text2wiki)
    #entities = text2wiki.keys()
    #wikis = text2wiki.values()
    normed_wikis = wiki_norm(wikis)

    textnode2index = {e:idx for idx,e in enumerate(entities)}
    wikinode2index = {w:idx for idx,w in enumerate(normed_wikis)}
    
    textG = json.loads(lines2[line_idx])
    wikiG = json.loads(lines3[line_idx])

    source_node_textG = []
    target_node_textG = []
    for pair in textG:
        source_node_textG.append(textnode2index[pair[0]])
        target_node_textG.append(textnode2index[pair[1]])
    need_textG = np.array([source_node_textG+target_node_textG, target_node_textG+source_node_textG])
    #print('need_textG', need_textG)

    source_node_wikiG = []
    target_node_wikiG = []
    for pair in wikiG:
        source_node_wikiG.append(wikinode2index[pair[0]])
        target_node_wikiG.append(wikinode2index[pair[1]])
    need_wikiG = np.array([source_node_wikiG+target_node_wikiG, target_node_wikiG+source_node_wikiG])
    #print('need_wikiG', need_wikiG)


    num_sentences = len(sentences)
    sentence_list = []
    entity_list = []

    for i,s in enumerate(sentences):
        sentence = s.lower()
        #print('sentence', sentence)
        for j,e in enumerate(entities):
            #print('entity', e)
            if re.search(e, sentence):
                sentence_list.append(i)
                entity_list.append(j) 
    #print('sen_list', sentence_list)
    #print('en_list', entity_list)

    #print('num_nodes', num_nodes)
    #print('num_sentence', num_sentences)
    sentence_entity_A = np.array([1]*len(sentence_list))

    return np.array(num_nodes), need_textG, need_wikiG, np.array(num_sentences), np.array(sentence_list), np.array(entity_list), sentence_entity_A


def gen_sample(save_path, start_index):
    label_dict = {'human': 0, 'machine': 1}
    f1.seek(0)
    for idx, line in enumerate(islice(f1, start_index, None)):
        real_idx = start_index + idx
        print('processing line {}...'.format(real_idx))
        
        text = json.loads(line)['article']
        text_emb = get_document_emb(text)

        sentences = text.strip('.').split('.')
        sentence_emb = get_sentence_emb(sentences)

        text2wiki = json.loads(lines4[idx])
        entities = text2wiki.keys()
        wikis = text2wiki.values()
        entities_emb = get_entity_emb_w2v(entities, unk)

        num_nodes, need_textG, need_wikiG, num_sentences, sentence_list, entity_list, sen_en_A_data = gen_3G(real_idx, text2wiki, entities, wikis, sentences)

        label = np.array(label_dict[json.loads(line)['label']])

        np.savez(os.path.join(save_path, '{}.npz').format(str(real_idx)), text_emb=text_emb, sentence_emb=sentence_emb, entities_emb=entities_emb,
                 num_nodes=num_nodes, textG_A=need_textG, wikiG_A=need_wikiG, num_sentences=num_sentences, sen_en_A_row=sentence_list,
                 sen_en_A_col=entity_list, sen_en_A_data=sen_en_A_data, label=label)
        
if __name__ == '__main__':
    f1 = open(text_path, 'r', encoding='utf8')
    f2 = open(textG_path, 'r', encoding='utf8')
    f3  = open(wikiG_path, 'r', encoding='utf8')
    f4 = open(text2wiki_path, 'r', encoding='utf8')

    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()
    lines4 = f4.readlines()

    save_path = '../data/grover/train_npz'  
    start_index = 0
    gen_sample(save_path, start_index)   

    f1.close()
    f2.close()
    f3.close()
    f4.close()