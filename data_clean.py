import os
import json
import re

data_name = 'grover'
data_path = os.path.join('../data', data_name, 'grover_10000_train.jsonl')
clean_data = os.path.join('../data', data_name, 'train_clean.jsonl')

with open(data_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f.readlines()):
        data = json.loads(line)
        #print(data)
        text = data['article']
        sentences = text.split('.')
        #title = data['title']
        #domain = data['domain']
        label = data['label']

        clean_text = []
        for i in range(len(sentences)):
            #print(sentences[i])
            clean_sentence = re.sub(r"[\n]+", " ", sentences[i]) 
            clean_sentence = re.sub(r"[^a-zA-Z0-9.!? ]+", "", clean_sentence)
            #clean_sentence = remove_stopwords(clean_sentence)
            #print(clean_sentence)
            clean_text.append(clean_sentence + '.')
            
        used_text = " ".join(clean_text)  

        new_data = {'article': used_text,
                    #'title': title,
                    #'domain': domain,
                    'label': label}
        
        with open(clean_data, 'a', encoding='utf8') as fw:
            json_str = json.dumps(new_data)
            fw.write(json_str + '\n')
