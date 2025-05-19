# needed to load the REBEL model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
import torch.nn as nn
import json
from itertools import islice
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import IPython

import tagme
tagme.GCUBE_TOKEN = "xxxxxxxx"
# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("../model/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("../model/rebel-large").to(device)
# model.to(device)
#extract entities and their relationships from text
#from https://huggingface.co/Babelscape/rebel-large
def extract_relations_from_model_output(text):
    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text_replaced.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


# knowledge base class
class KB():
    def __init__(self):
        self.relations = []

    def are_relations_equal(self, r1, r2):
        return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

    def exists_relation(self, r1):
        return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

    def merge_relations(self, r1):
        r2 = [r for r in self.relations
              if self.are_relations_equal(r1, r)][0]
        spans_to_add = [span for span in r1["meta"]["spans"]
                        if span not in r2["meta"]["spans"]]
        r2["meta"]["spans"] += spans_to_add

    def add_relation(self, r):
        # manage new relation
        if not self.exists_relation(r):
            self.relations.append(r)
        else:
            self.merge_relations(r) 

    def print(self):
        print("Relations:")
        for r in self.relations:
            print(f"  {r}")


# extract relations for each span and put them together in a knowledge base
def from_text_to_kb(text, span_length=64, verbose=False):
    # tokenize whole text
    inputs = tokenizer([text], return_tensors="pt")

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    #if verbose:
    #    print(f"Input has {num_tokens} tokens")
    num_spans = math.ceil(num_tokens / span_length)
    #if verbose:
    #    print(f"Input has {num_spans} spans")
    overlap = math.ceil((num_spans * span_length - num_tokens) / 
                        max(num_spans - 1, 1))
    spans_boundaries = []
    start = 0
    for i in range(num_spans):
        spans_boundaries.append([start + span_length * i,
                                 start + span_length * (i + 1)])
        start -= overlap
    #if verbose:
    #    print(f"Span boundaries are {spans_boundaries}")

    # transform input with spans
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]]
                  for boundary in spans_boundaries]
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]]
                    for boundary in spans_boundaries]
    inputs = {
        "input_ids": torch.stack(tensor_ids).to(device),
        "attention_mask": torch.stack(tensor_masks).to(device)
    }

    # generate relations
    num_return_sequences = 3
    gen_kwargs = {
        "max_length": 512,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    generated_tokens = model.generate(
        **inputs,
        **gen_kwargs,
    )

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens,
                                           skip_special_tokens=False)

    # create kb
    kb = KB()
    i = 0
    for sentence_pred in decoded_preds:
        current_span_index = i // num_return_sequences
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "spans": [spans_boundaries[current_span_index]]
            }
            #print(relation)
            kb.add_relation(relation)
        i += 1

    return kb


#entract entities from KB of text, noted Gini
def get_Gini(kb):
    Gini= []
    for r in kb.relations:
        s_entity = r['head'].lower()
        d_entity = r['tail'].lower()
        Gini.append(s_entity)
        Gini.append(d_entity)
    return list(set(Gini))


#entity linking
def Annotate(txt, theta=0.1):
    """
    :param txt: str
    :param language: defualt ''en''
    :param theta:[0, 1], the score, default 0.1
    :return: [(A, B):score]
    """
    dic = dict()
    try:
        annotations = tagme.annotate(txt)
        for ann in annotations.get_annotations(theta):
            # print(ann)
            try:
                A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                dic[(A, B)] = score
            except:
                print('error')
    except:
        print('error')
    return dic

#select linked entities when theta is bigger than 0.2 and the biggest one if the number of linked entities is more than 2
def get_text2wiki(kb, text_entities):
    #text_entities = get_Gini(kb)
    #print('text_entities',text_entities)
    text2wiki = {}
    text_index = []
    for i, entity in enumerate(text_entities):
        if len(entity) == 0:
            continue
        else:
            obj = Annotate(str(entity), theta=0.2)
            if len(obj) == 1:
                keys = list(obj.keys())
                #print(keys)
                text2wiki[entity] = keys[0][1]
                text_index.append(i)
            elif len(obj) > 1:
                best_key = ''
                best_value = '0'
                for k,v in obj.items():
                    if float(v) > float(best_value):
                        best_key = k
                        best_value = v
                text2wiki[entity] = best_key[1]
                text_index.append(i)
            else:
                pass
    #print('text2wiki',text2wiki)
    #print(text_index)
    return text2wiki, text_index

#generate entity graph of a text, where entities are linked to wikipedia
def gen_text_graph(kb, text_entities, text_index):
    text_G_ini = []
    for r in kb.relations:
        s_entity = r['head'].lower()
        d_entity = r['tail'].lower()
        text_G_ini.append((s_entity, d_entity))
    #print('text_G_ini', text_G_ini)

    text_en_in_wiki = []
    for idx in text_index:
        text_entity = text_entities[idx]
        text_en_in_wiki.append(text_entity)
    #print('text_en_in_wiki', text_en_in_wiki)

    text_G_need = []
    for pair in text_G_ini:
        flag = all(entity in text_en_in_wiki for entity in list(pair))
        if flag:
            #if pair[0] > pair[1]:
            text_G_need.append(pair)
    #print('text_G_need', set(text_G_need))

    text_G_need_norm = set()
    for pair in text_G_need:
        if pair[0] > pair[1]:
            text_G_need_norm.add((pair[1], pair[0]))
        elif pair[1] < pair[0]:
            text_G_need_norm.add((pair[0], pair[1]))
        else:
            text_G_need_norm.add(pair)

    #print('set text_G_need_norm', text_G_need_norm)
    return list(text_G_need_norm)


if __name__ == "__main__":
    data_path = '../data/grover/train_clean.jsonl'
    text2wiki_path = '../data/grover/train_clean_text2wiki.jsonl'
    textG_path = '../data/grover/train_clean_textG.jsonl'
    print(model.device)
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        start_index = 0
        f.seek(0)
        for line_num, line in enumerate(islice(f, start_index, None)):

            print("processing line {}...".format(start_index+line_num))
            data = json.loads(line)
            #print(data)
            text = data['article'].split()
            text = ' '.join(text[:2000])
            #print('text', text)
            kb = from_text_to_kb(text, verbose=True)    
            #print('kb', kb)
            text_entities = get_Gini(kb)
            #print('text_entities', text_entities)
            try:
                if len(text_entities) == 0:
            #         print("The number of entities in line {} is zero.".format(start_index+line_num))
                    with open(text2wiki_path, 'a', encoding='utf8') as fwiki:
                        json_str = json.dumps('{}')
                        fwiki.write(json_str + '\n') 
                    with open(textG_path, 'a', encoding='utf8') as fg:
                        json_g = json.dumps('[]')
                        fg.write(json_g + '\n') 

                else:
                    text2wiki, text_index = get_text2wiki(kb, text_entities)
                    with open(text2wiki_path, 'a', encoding='utf8') as fwiki:
                        json_str = json.dumps(text2wiki)
                        fwiki.write(json_str + '\n')

                    text_G = gen_text_graph(kb, text_entities, text_index)
                    with open(textG_path, 'a', encoding='utf8') as fg:
                        json_g = json.dumps(text_G)
                        fg.write(json_g + '\n')
            except Exception as e:
                print(e)