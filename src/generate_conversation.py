
# coding: utf-8

# In[1]:


import argparse
import json
import logging
import os
import re
import nltk
import torch
import random

from torch.autograd import Variable
from vocabulary import Vocabulary

logger=logging.getLogger(__name__)


max_len=20

class Raw_Sent:
    
    def __init__(self):
        self.index2sent={}
        
        
    def __len__(self):
        return len(index2sent)


    def add_sent(self,sentence,index):
        #print(index)
        self.index2sent[index]=sentence

    def get_sent(self,index):
        return self.index2sent[index]

class Dialogue_Info:
    
    def __init__(self):
        self.utterance=Raw_Sent()
        self.dial_info=[]


    def normalize_string(self,s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?\']+", r" ", s)
        return s


    def preprocess_utter(self,line):
        
        line=line.split('+++$+++')
        index=nltk.word_tokenize(line[0])[0]
        
        #self.vocab.init_vocab(line[4])
        
        utterance=self.normalize_string(line[4])

        #utterance=nltk.word_tokenize(utterance)

        return index,utterance

    
    def preprocess_info(self,line):
        
        line=line.replace(' ','').split('+++$+++')

        line[3]=line[3].replace('\n','').replace('\'','"')
        info=json.loads(line[3])
        return info

    def save_dialogue_info(self,data_dir):
        path=os.path.join(data_dir,'dialogue_info.json')
        with open(path,'w') as f:
            json.dump({
                'utterance':self.utterance.index2sent,
                'dial_data':self.dial_info},f)
            
    def load_dialogue_info(self,data_dir):
        path=os.path.join(data_dir,'dialogue_info.json')
        with open(path,'r') as f:
            data=json.load(f)
            
        self.utterance=data['utterance']
        self.dial_info=data['dial_data']


    def build_dialogue_info(self, corpus_path):
        dial_utter_path=os.path.join(corpus_path,'movie_lines.txt')
        dial_info_path=os.path.join(corpus_path,'movie_conversations.txt')
        
        
        with open(dial_utter_path,'r',errors='ignore') as utter_file:
            for line in utter_file:
                index,utterance=self.preprocess_utter(line)
                self.utterance.add_sent(utterance,index)
                
        with open(dial_info_path,'r') as info_file:
            for line in info_file:
                info=self.preprocess_info(line)
                self.dial_info.append(info)

#		self.save_dialog_data(dialog_path)



class Conversation:
    
    def __init__(self ):
        self.dial_info=Dialogue_Info()
        self.vocab=Vocabulary()
        
        self.dial_pair=[]
        self.train_dial_pair=[]
        self.valid_dial_pair=[]
        self.test_dial_pair=[]
        
        self.encoded_train_dial_pair=[]
        self.encoded_valid_dial_pair=[]
        self.encoded_test_dial_pair=[]
        
    
    def split_data_set(self,train_ratio=0.8,valid_ratio=0.1,test_ratio=0.1):
        
        train_size=int(len(self.dial_pair)*train_ratio)
        valid_size=int(len(self.dial_pair)*valid_ratio)
        test_size=int(len(self.dial_pair)*test_ratio)
        
        while len(self.valid_dial_pair)<valid_size:
            index=random.randint(0,len(self.dial_pair)-1)
            if index not in self.valid_dial_pair:
                self.valid_dial_pair.append(self.dial_pair[index])
                
        while len(self.test_dial_pair)<test_size:
            index=random.randint(0,len(self.dial_pair)-1)
            if index not in self.test_dial_pair and index not in self.valid_dial_pair:
                self.test_dial_pair.append(self.dial_pair[index])
                
        for index in range(len(self.dial_pair)):
            if index not in self.valid_dial_pair and index not in self.test_dial_pair:
                self.train_dial_pair.append(self.dial_pair[index])
                if len(self.train_dial_pair)>=train_size: break
    
    def build_dialogue_pair(self):
        
        for item in self.dial_info.dial_info:
            item_len=len(item)
            if item_len%2 !=0:
                item=item[0:int(item_len/2)*2]
                
            for index in range(0,len(item),2):
                self.dial_pair.append(list((item[index],item[index+1])))
                

    def encode_dialogue_pair(self):
        #self.get_raw_sent(source_path)
        
        for train_item in self.train_dial_pair:
            if len(train_item[0])>1 and len(train_item[0])<max_len and                 len(train_item[1])>1 and len(train_item[1])<max_len :
                self.encoded_train_dial_pair.append([self.vocab.encode_sent(train_item[0]),self.vocab.encode_sent(train_item[1])])
                
        for valid_item in self.valid_dial_pair:
            if len(valid_item[0])>1 and len(valid_item[0])<max_len and                 len(valid_item[1])>1 and len(valid_item[1])<max_len :
                self.encoded_valid_dial_pair.append([self.vocab.encode_sent(valid_item[0]),self.vocab.encode_sent(valid_item[1])])
                
                
        for test_item in self.test_dial_pair:
            if len(test_item[0])>1 and len(test_item[0])<max_len and                 len(test_item[1])>1 and len(test_item[1])<max_len :
                self.encoded_test_dial_pair.append([self.vocab.encode_sent(test_item[0]),self.vocab.encode_sent(test_item[1])])


    def save_dialogue_pair(self,exp_data_dir):
        dial_pair_path=os.path.join(exp_data_dir,'dialogue_pair.json')
        train_dial_pair_path=os.path.join(exp_data_dir,'train_dialogue_pair.json')
        valid_dial_pair_path=os.path.join(exp_data_dir,'valid_dialogue_pair.json')
        test_dial_pair_path=os.path.join(exp_data_dir,'test_dialogue_pair')
        
        encoded_train_dial_pair_path=os.path.join(exp_data_dir,'encoded_train_dialogue_pair.json')
        encoded_valid_dial_pair_path=os.path.join(exp_data_dir,'encoded_valid_dialogue_pair.json')
        encoded_test_dial_pair_path=os.path.join(exp_data_dir,'encoded_test_dialogue_pair.json')
        
        with open(dial_pair_path,'w') as f:
            json.dump(self.dial_pair,f)
            
        with open(train_dial_pair_path,'w') as f:
            json.dump(self.train_dial_pair,f)
            
            
        with open(valid_dial_pair_path,'w') as f:
            json.dump(self.valid_dial_pair,f)
            
            
        with open(test_dial_pair_path,'w') as f:
            json.dump(self.test_dial_pair,f)
            
        with open(encoded_train_dial_pair_path,'w') as f:
            json.dump(self.encoded_train_dial_pair,f)
            
        with open(encoded_valid_dial_pair_path,'w') as f:
            json.dump(self.encoded_valid_dial_pair,f)

        with open(encoded_test_dial_pair_path,'w') as f:
            json.dump(self.encoded_test_dial_pair,f)
            
            
    def create_conversation(self,raw_data_dir,exp_data_dir):
        threshold=5
        
        self.dial_info.build_dialogue_info(raw_data_dir)
        self.dial_info.save_dialogue_info(exp_data_dir)
        
        self.vocab.build_vocab(threshold,self.train_dial_pair)
        self.vocab.save_vocab(exp_data_dir)
        
        self.build_dialogue_pair()
        self.split_data_set()
        self.encode_dialogue_pair()
        self.save_dialogue_pair(exp_data_dir)
        
        


# In[2]:

if __name__=='__main__':
    conv=Conversation()
    conv.create_conversation('../raw_data','../experiment_data')

