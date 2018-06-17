
# coding: utf-8

# In[ ]:


import json
import logging
import re
import os
import sys
import nltk


class Vocabulary:
    
    def __init__(self):
        
        self.word2idx={"SOS": 0,"EOS":1,"PADED":2,"UNK":3}
        self.idx2word={0:"SOS",1:"EOS",2:"PADED",3:"UNK"}
        self.word2count={}
        
    def get_vocab_size(self):
        return len(self.word2idx)
    
    def add_words(self,word):
        #print(word)
        if word not in self.word2count:
            self.word2count[word]=1
        else:
            self.word2count[word]=self.word2count[word]+1
            #print(self.word2count)
            
    def add_word2vocab(self,word):
        if word not in self.word2idx:
            self.idx2word[len(self.word2idx)]=word
            self.word2idx[word]=len(self.word2idx)
            
    def get_idx(self,word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['UNK']
        
    def get_word(self,idx):
        if str(idx) in self.idx2word:
            return self.idx2word[str(idx)]
        else:
            return 'UNK'
        
    def get_word_count(self,word):
        return self.word2count[word]
    
    def sort_word_freq(self):
        sorted_vocab=sorted(self.word2count.items(),key=lambda d:d[1],reverse=True)
        self.word2count={}
        for item in sorted_vocab:
            self.word2count[item[0]]=item[1]
            
    def update_vocab(self,threshold,freq=True):
        #print(self.word2count)
        for index,item in enumerate(self.word2count.items()):
            self.add_word2vocab(item[0])
            #print(item)
            if not freq:
                if len(self.word2idx)>=threshold: return
            else:
                if item[1]<threshold: return


    def save_vocab(self,vocab_file):
        with open(vocab_file,'w') as f:
            json.dump({
                'word2idx':self.word2idx,
                'idx2word':self.idx2word,
                'word2count':self.word2count},f)
            
    def load_vocab(self,vocab_file):
        with open(vocab_file,'r') as f:
            vocab=json.load(f)
        self.word2idx=vocab['word2idx']
        self.idx2word=vocab['idx2word']
        self.word2count=vocab['word2count']
        
    def init_vocab(self,sentence):
        word_list=nltk.word_tokenize(sentence)
        for token in word_list:
            self.add_words(token)
            
            
    def build_vocab(self,threshold,data_set=None,path=None,freq=True):
        if data_set==None and path !=None:
            with open(path,'r') as f:
                data_set=json.load(f)
                
        if data_set==None and path ==None:
            return 
        
        for item in data_set:
            for sent in item:
                self.init_vocab(sent)
            
        self.sort_word_freq()
        self.update_vocab(threshold,freq)
        
        
    def encode_sent(self,sent):
        sent=nltk.word_tokenize(sent)
        encoded_sent=[self.get_idx(word) for word in sent]
        encoded_sent.append(self.get_idx("EOS"))
        return encoded_sent

    @classmethod
    def seq_pad(cls,seq,max_len):
        vocab=Vocabulary()
        seq+=[vocab.get_idx('PADED') for i in range(max_len-len(seq))]
        return seq
    
    def decode_sent(self,sent):
        deco_sent=''
        for word in sent:
            if word != self.get_idx('EOS'):
                deco_sent+=self.get_word(word)
                deco_sent+=' '
            else:
                break
        return deco_sent.strip()
    
    
    def test(self):
        os.path.join('raw_data','file.json')


# In[ ]:


if __name__=='__main__':
    data_dir='../experiment_data/data'
    vocab=Vocabulary()
    vocab.build_vocab(5,data_set=None,path=os.path.join(data_dir,'train_dialogue_pair.json'))
    vocab.save_vocab(os.path.join(data_dir,'vocabulary.json'))

