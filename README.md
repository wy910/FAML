# Non-Existent Relationship: Fact-Aware Multi-Level Machine-Generated Text Detection

## Environment Setup
```bash
# python==3.8.10
pip install -r requirements.txt
```

## Pre-processing
The GROVER and GPT-2 public datasets are at https://huggingface.co/datasets/ZachW/MGTDetect_CoCo. The SemEval dataset is at https://github.com/mbzuai-nlp/SemEval2024-task8. You need to download the three datasets to your localhost 'data' folder.
Put the 'code' and 'data' folders in the same directory.

You need to build the Dbpedia knowledge base locally.

```bash
# Data cleaning
python3 data_clean.py
# Get text2wiki map and textual entity graph
python3 data_process.py
# Get factual entity graph
python3 get_wikiG.py
# Preparing data for model input
python3 data_ready.py
```

## Train, validate, and test models
After processing all the data in the three datasets, you can proceed with training and testing.
```bash
python3 main.py

```
