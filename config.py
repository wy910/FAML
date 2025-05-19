# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch


parser = argparse.ArgumentParser(description='HG')

parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gcn_lr', type=float, default=1e-5)
parser.add_argument('--senG_lr', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)  #0.5
parser.add_argument('--w_decay', type=float, default=0.001)  #0.001

parser.add_argument('--model_name', type=str, default='FAML')
parser.add_argument('--dataset', type=str, default='grover')
parser.add_argument('--save', type=str, default='../save/grover', help='folder to save the final model')
parser.add_argument('--model_path', type=str, default='../model/RoBERTa-large')
parser.add_argument('--entity_dim', type=int, default=300)
parser.add_argument('--text_in_dim', type=int, default=1024)  
parser.add_argument('--sen_in_dim', type=int, default=1024) 
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=128)  
parser.add_argument('--n_layers', type=int, default=6) 
parser.add_argument('--sen_max_len', type=int, default=1200)
parser.add_argument('--en_max_len', type=int, default=200)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=256)   

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')