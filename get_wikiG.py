from SPARQLWrapper import SPARQLWrapper, JSON
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from itertools import islice
import time
import re 

# DBpedia SPARQL endpoint
sparql = SPARQLWrapper("http://localhost:8890/sparql")

def query_with_retry(query, retries=3):
    """Query function with retry mechanism"""
    for attempt in range(retries):
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            return sparql.query().convert()
        except Exception as e:
            print(f"Query failed (attempt {attempt + 1}): {e}")
            time.sleep(2)  # Delay to avoid frequent requests
    raise Exception("Max retries exceeded")

# SPARQL template to query whether an entity exists
def check_entity_exists(entity):
    #print('entity', entity)
    query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    ASK WHERE {{
        dbr:{entity} ?p ?o .
    }}
    """
    #sparql.setQuery(query)
    #sparql.setReturnFormat(JSON)
    #result = sparql.query().convert()
    result = query_with_retry(query)
    return result['boolean']

# SPARQL template for querying the relationship between two entities
def check_entity_relationship(entity1, entity2):
    query = f"""
    PREFIX dbr: <http://dbpedia.org/resource/>
    SELECT ?property WHERE {{
        dbr:{entity1} ?property dbr:{entity2} .
    }} LIMIT 5
    """ 
    #sparql.setQuery(query)
    #sparql.setReturnFormat(JSON)
    #results = sparql.query().convert()
    results = query_with_retry(query)
    return [result["property"]["value"] for result in results["results"]["bindings"]]


def gen_wiki_graph(text2wiki):
    wiki_G = []
    wiki_en = []
    for key, value in text2wiki.items():
        wiki_en.append(value)
    wiki_en = list(set(wiki_en))
    wiki_en_norm = []
    for e in wiki_en:
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
    #print('wiki_entities', wiki_en_norm)

    #return []

    wiki_en_nofind = []
    wiki_en_find = []
    for i in range(len(wiki_en_norm)):
        exit_flag = check_entity_exists(wiki_en_norm[i])
        if exit_flag:
            wiki_en_find.append(wiki_en_norm[i])
        else:
            wiki_en_nofind.append(wiki_en_norm[i])
            #print(f"Entity {wiki_en[i]} does not exist in DBpedia.")

    #find pairs
    for i in range(len(wiki_en_find)):
        for j in range(i + 1, len(wiki_en_find)):
            relationships1 = check_entity_relationship(wiki_en_find[i], wiki_en_find[j])
            relationships2 = check_entity_relationship(wiki_en_find[j], wiki_en_find[i])
            if relationships1 or relationships2:
                wiki_G.append((wiki_en_find[i], wiki_en_find[j]))
            else:
                pass
                #print(f"{wiki_en_find[i]} and {wiki_en_find[j]} have no any relationships.")

    return list(wiki_G)


if __name__ == '__main__':
    fold = 'grover'
    text2wiki_file = os.path.join('../data', fold, 'train_clean_text2wiki.jsonl')
    save_file = os.path.join('../data', fold, 'train_clean_wikiG.jsonl')
    
    with open(text2wiki_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        start_index = 0
        f.seek(0)
        for line_num, line in enumerate(islice(f, start_index, None)):
            print("processing line {}...".format(start_index+line_num))
            data = json.loads(line)
            wikiG = gen_wiki_graph(data)
            #print('wikiG', wikiG)
            with open(save_file, 'a', encoding='utf8') as fw:
                json_g = json.dumps(wikiG)   
                fw.write(json_g + '\n') 
