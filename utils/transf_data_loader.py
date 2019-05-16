import sys
sys.path.append('../')
import json
import os
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable
from pytorch_pretrained_bert import TransfoXLTokenizer
from utils.data_loader import FileDataLoader

class JSONFileDataLoaderTransf(FileDataLoader):
    def __init__(self, file_name, max_length=40, case_sensitive=False, cuda=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                                "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177": 
                    [
                        ...
                    ]
                ...
            }
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda
        
        # Check files
        if file_name is None or not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        # Load files
        print("Loading data file...")
        self.ori_data = json.load(open(self.file_name, "r"))
        print("Finish loading")
            
        # Eliminate case sensitive
        if not case_sensitive:
            print("Elimiating case sensitive problem...")
            for relation in self.ori_data:
                for ins in self.ori_data[relation]:
                    for i in range(len(ins['tokens'])):
                        ins['tokens'][i] = ins['tokens'][i].lower()
            print("Finish eliminating")
 
        # Init tokenizer
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        self.tokenizer.add_symbol('<sep1>')
        self.tokenizer.add_symbol('<sep2>')
        self.tokenizer.add_symbol('<cls>')
        self.tokenizer.add_symbol('<blk>')
        self.tokenizer.add_symbol('<start>')
        UNK = self.tokenizer.convert_tokens_to_ids(['<unk>'])[0]
        BLANK = self.tokenizer.convert_tokens_to_ids(['<blk>'])[0]
        CLS = self.tokenizer.convert_tokens_to_ids(['<cls>'])[0]
        SEP1 = self.tokenizer.convert_tokens_to_ids(['<sep1>'])[0]
        SEP2 = self.tokenizer.convert_tokens_to_ids(['<sep2>'])[0]
        START = self.tokenizer.convert_tokens_to_ids(['<start>'])[0]
            
        print("Finish building")

        # Pre-process data
        print("Pre-processing data...")
        self.instance_tot = 0
        for relation in self.ori_data:
            self.instance_tot += len(self.ori_data[relation])

        self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
        self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
        self.rel2scope = {} # left close right open
            
        i = 0
        for relation in self.ori_data:
            self.rel2scope[relation] = [i, i]
            for ins in self.ori_data[relation]:
                head_indices = ins['h'][2][0]
                tail_indices = ins['t'][2][0]
                words = ins['tokens']
                
                word_indices = self.tokenizer.convert_tokens_to_ids(words)
                curr_list = [START] + head_indices + [SEP1] + tail_indices + \
                            [SEP2] + word_indices
                
                curr_list = curr_list[:self.max_length]
                
                self.data_length[i] = len(curr_list)
                
                while len(curr_list) < self.max_length:
                    curr_list.append(BLANK)
                curr_list[-1] = CLS
                
                self.data_word[i] = np.array(curr_list)
                    
                i += 1
            self.rel2scope[relation][1] = i 

        print("Finish pre-processing")

    def next_one(self, N, K, Q):
        target_classes = random.sample(self.rel2scope.keys(), N)
        support_set = []
        query_set = []
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_set.append(support_word)
            query_set.append(query_word)
            query_label += [i] * Q

        support_set = np.stack(support_set, 0)
        query_set = np.concatenate(query_set, 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q)
        query_set = query_set[perm]
        query_label = query_label[perm]

        return support_set, query_set, query_label

    def next_batch(self, B, N, K, Q):
        support = []
        query = []
        label = []
        
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(N, K, Q)
            support.append(current_support)
            query.append(current_query)
            label.append(current_label)
            
        support = Variable(torch.from_numpy(np.stack(support, 0)).long().view(-1, self.max_length))
        query = Variable(torch.from_numpy(np.stack(query, 0)).long().view(-1, self.max_length))  
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())
        
        # To cuda
        if self.cuda:
            support = support.cuda()
            query = query.cuda()
            label = label.cuda()

        return support, query, label