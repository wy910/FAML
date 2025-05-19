# -*- coding: utf-8 -*-

from config import args
from dataset import *
from model import *
from earlystopping import *

from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader
#from torch.amp import GradScaler, autocast
from torch.optim import Adam, lr_scheduler
import torch.optim as optim
from sklearn.metrics import classification_report

import random
from tqdm import tqdm  
import matplotlib.pyplot as plt
import json
import time

def set_seeds(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def training(args):
    print('\n{} Experimental Dataset: {} {}\n'.format(
        '=' * 20, args.dataset, '=' * 20))
    #print('save path: ', args.save)
    print('Start time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    #print('Loading data...')
    start = time.time()

    dataset_name = args.dataset
    train_text_path = os.path.join('../data', dataset_name, 'train_clean.jsonl')
    val_text_path = os.path.join('../data', dataset_name, 'dev_clean.jsonl')
    with open(train_text_path, 'r', encoding='utf8') as ft:
        train_text = ft.readlines()

    with open(val_text_path, 'r', encoding='utf8') as fv:
        val_text = fv.readlines()

    train_dataset = HGDataset(args, train_text, 'train_npz')
    val_dataset = HGDataset(args, val_text, 'dev_npz')

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        #pin_memory=(torch.cuda.is_available()),

        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        #pin_memory=(torch.cuda.is_available()),
        drop_last=True,
        sampler=val_sampler
    )

    #print('Loading data time: {:.2f}s\n'.format(time.time() - start))
    #print('-----------------------------------------\nLoading model...\n')
    start = time.time()
    model = HGmodel(args).to(args.device)

    criterion = nn.CrossEntropyLoss().cuda()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.w_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
    ]

    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.w_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 

    #training
    # Best results on validation dataset
    best_val_result = 0
    best_val_epoch = -1
    best_epoch = 0
    start = time.time()

    early_stop = EarlyStopping()

    for epoch in range(args.epochs):
        print('\n------------------------------------------------\n')
        print('Start Training Epoch', epoch, ':', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        
        model.train()

        train_loss = 0.
        print_step = 10

        for step, train_data in enumerate(tqdm(train_loader)):
            train_report_label = []
            train_report_predict = []

            train_data.to(args.device)
            train_label = train_data['y']#.to(args.device)

            optimizer.zero_grad()

            #with autocast():
            out, pred = model(train_data)
            loss = criterion(out, train_label)
            if torch.any(torch.isnan(loss)):
                print('out: ', out)
                print('loss = {:.4f}\n'.format(loss.item()))
                exit()

            train_label_np = train_label.cpu().detach().numpy()
            train_predict = torch.max(pred.cpu().detach(), 1)[1]
                
            for i in range(len(train_label_np)):
                train_report_label.append(train_label_np[i])
                train_report_predict.append(train_predict[i])

            train_report = classification_report(train_report_label, train_report_predict, output_dict = True)
            train_acc = train_report['accuracy']
            train_f1 = float(train_report["macro avg"]["f1-score"])

            if step % print_step == 0:
                print('\n\nEpoch: {}, Step: {}, Loss = {:.4f}, Acc = {}, F1 = {}'.format(epoch, step, loss.item(), train_acc, train_f1))
                
            # Gradient cropping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        #evaluating
        model.eval()
        eval_loss = 0.
        report_label = []
        report_predict = []

        with torch.no_grad():
            for val_data in tqdm(val_loader):
                val_data.to(args.device)
                val_label = val_data['y']#.to(args.device)

                val_out, val_pred = model(val_data)
                loss = criterion(val_out, val_label)
                eval_loss += loss.item()

                val_label_np = val_label.cpu().detach().numpy()
                predict = torch.max(val_pred.cpu().detach(), 1)[1]
                
                for i in range(len(val_label_np)):
                    report_label.append(val_label_np[i])
                    report_predict.append(predict[i])

        eval_loss /= len(val_loader)

        report = classification_report(report_label, report_predict, output_dict = True)
        val_acc = report['accuracy']
        val_f1 = float(report["macro avg"]["f1-score"])


        print('='*10, 'Epoch: {}/{}'.format(epoch, args.epochs), '='*10)
        print('\n[Loss]\nTrain: {:.6f}\tVal: {:.6f}\n'.format(train_loss, eval_loss))
        print('-'*10)
        print('\n[Macro F1] Acc: {:.6f}\n'.format(val_acc))
        print('\n[Macro F1] Val: {:.6f}\n'.format(val_f1))
        print('-'*10)

        if val_f1 >= best_val_result:
            best_val_result = val_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                os.path.join(args.save, '{}_epoch_{}_lr_{}_ba_{}_dr_{}_wd_{}_hd_{}_od_{}_n_{}_h_{}_df_{}_val_{}.pt'.format(args.model_name, best_epoch, args.lr, args.batch_size, args.dropout, args.w_decay, args.hid_dim, args.out_dim, 
                                                                                                                        args.n_layers, args.heads, args.d_ff, best_val_result))
            )
        
        early_stop(eval_loss)
        if early_stop.early_stop:
            print('Early_stopping')
            break
        

    print('Training Time: {:.2f}s'.format(time.time() - start))
    print('End time:', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    return os.path.join(args.save, '{}_epoch_{}_lr_{}_ba_{}_dr_{}_wd_{}_hd_{}_od_{}_n_{}_h_{}_df_{}_val_{}.pt'.format(args.model_name, best_epoch, args.lr, args.batch_size, args.dropout, args.w_decay, args.hid_dim, args.out_dim, 
                                                                                                                        args.n_layers, args.heads, args.d_ff, best_val_result))


def test(args, model_path):
    dataset_name = args.dataset
    test_text_path = os.path.join('../data', dataset_name, 'test_clean.jsonl')
    with open(test_text_path, 'r', encoding='utf8') as ft:
        test_text = ft.readlines()
    
    test_dataset = HGDataset(args, test_text, 'test_npz')
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        #pin_memory=(torch.cuda.is_available()),
        drop_last=False,
        sampler=test_sampler
    )

    model = HGmodel(args).to(args.device)
    #model = DataParallel(model)
    model.load_state_dict(torch.load(model_path)['state_dict'])

    criterion = nn.CrossEntropyLoss().cuda()

    model.eval()
    test_loss = 0.
    report_label = []
    report_predict = []

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_data.to(args.device)
            test_label = test_data['y']
            
            test_out, test_pred = model(test_data)
            loss = criterion(test_out, test_label)
            test_loss += loss.item()

            test_label_np = test_label.cpu().detach().numpy()
            predict = torch.max(test_pred.cpu().detach(), 1)[1]
                
            for i in range(len(test_label_np)):
                report_label.append(test_label_np[i])
                report_predict.append(predict[i])

    test_loss /= len(test_loader)

    report = classification_report(report_label, report_predict, output_dict = True)
    print(report)   
    acc = report['accuracy']
    f1 = report['macro avg']['f1-score']
    f1r = report['1']['f1-score']
    f1nr = report['0']['f1-score']
    return acc, f1, f1r, f1nr



if __name__ == '__main__':
    seed_list = [42, 43, 44, 45, 46]
    acc_list = []
    f1_list = []
    f1ai_list = []
    f1human_list = []
    for seed in seed_list:
        set_seeds(seed)
        model_path = training(args)
        acc, f1, f1ai, f1human = test(args, model_path)
        #print('acc:{}, f1:{}, f1ai:{}, f1human:{}'.format(acc, f1, f1ai, f1human))
        acc_list.append(acc)
        f1_list.append(f1)
        f1ai_list.append(f1ai)
        f1human_list.append(f1human)

    print('avg acc:{}, f1:{}, f1ai:{}, f1human:{}'.format(sum(acc_list)/len(acc_list), sum(f1_list)/len(f1_list), sum(f1ai_list)/len(f1ai_list), sum(f1human_list)/len(f1human_list)))