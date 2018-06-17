import argparse
import json
import logging
import os
import re
import spacy
import nltk
import torch

from torch.autograd import Variable

logger=logging.getLogger(__name__)


max_len=20

class Dictionary:

	def __init__(self):
		self.index2sent={}

	def __len__(self):
		return len(index2sent)


	def add_sent(self,sentence,index):
#		print(index)
		self.index2sent[index]=sentence

	def get_sent(self,index):
		return self.index2sent[index]




class Vocabulary:

	def __init__(self):

		self.word2idx={"SOS": 0,"EOS":1,"PADED":2,"UNK":3}
		self.idx2word={0:"SOS",1:"EOS",2:"PADED",3:"UNK"}
		self.word2count={}

	def get_vocab_size(self):
		return len(self.word2idx)


	def add_words(self,word):
#		print(word)
		if word not in self.word2count:
			self.word2count[word]=1

		else:
			self.word2count[word]=self.word2count[word]+1
#		print(self.word2count)

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
#		print(self.word2count)
		for index,item in enumerate(self.word2count.items()):
			self.add_word2vocab(item[0])
#			print(item)
			if not freq:
				if len(self.word2idx)>=threshold: return
			else:
				if item[1]<threshold: return


	def save_vocab(self,path):
		with open(path,'w') as f:
			json.dump({
					'word2idx':self.word2idx,
					'idx2word':self.idx2word,
					'word2count':self.word2count},f)


	def load_vocab(self,path):
		with open(path,'r') as f:
			vocab=json.load(f)
		self.word2idx=vocab['word2idx']
		self.idx2word=vocab['idx2word']
		self.word2count=vocab['word2count']



	def init_vocab(self,sentence):
		tokens=nltk.word_tokenize(sentence)
		for token in tokens:
			self.add_words(token)


	def build_vocab(self,path,threshold,freq=True):
		with open(path,'r') as f:
			dialog=json.load(f)['utterance']
		for item in dialog.items():
			self.init_vocab(item[1])
		self.sort_word_freq()
		self.update_vocab(threshold,freq)



class Conversation:

	def __init__(self):
		self.utterance=Dictionary()
		self.dial_data=[]


	def normalizeString(self,s):
		s = s.lower().strip()
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?\']+", r" ", s)
		return s


	def preprocess_utter(self,line):

		line=line.split('+++$+++')
		index=nltk.word_tokenize(line[0])[0]
#		self.vocab.init_vocab(line[4])
		utterance=self.normalizeString(line[4])
#		utterance=nltk.word_tokenize(utterance)
		return index,utterance
		
		
	def preprocess_info(self,line):

		line=line.replace(' ','').split('+++$+++')

		line[3]=line[3].replace('\n','').replace('\'','"')
		info=json.loads(line[3])
		return info	

	def save_dialog_data(self,path):
		with open(path,'w') as f:
			json.dump({
					'utterance':self.utterance.index2sent,
					'dial_data':self.dial_data},f)

	def load_dialog_data(self,path):
		with open(path,'r') as f:
			data=json.load(f)
		self.utterance=data['utterance']
		self.dial_data=data['dial_data']


	def build_dialog_data(self, corpus_path):
		dial_utter_path=os.path.join(corpus_path,'movie_lines.txt')
		dial_info_path=os.path.join(corpus_path,'movie_conversations.txt')


		with open(dial_utter_path,'r',errors='ignore') as utter_file:

			for line in utter_file:
				index,utterance=self.preprocess_utter(line)
				self.utterance.add_sent(utterance,index)	

		with open(dial_info_path,'r') as info_file:
			for line in info_file:
				info=self.preprocess_info(line)
				self.dial_data.append(info)

#		self.save_dialog_data(dialog_path)



class DialPair:

	def __init__(self,corpus_path,dialog_path,vocab_path,vocab_threshold,is_freq):
		self.dialog=Conversation()
		self.vocab=Vocabulary()
		self.dialog_pair=[]
		self.encoded_dial_pair=[]

		self.dialog.build_dialog_data(corpus_path)
		self.dialog.save_dialog_data(dialog_path)
		self.vocab.build_vocab(dialog_path,vocab_threshold,is_freq)
		self.vocab.save_vocab(vocab_path)

	def encode_sent(self,sent):
		word_list=nltk.word_tokenize(sent)
		encode_words=list()
		for word in word_list:
			encode_words.append(self.vocab.get_idx(word))
		encode_words.append(self.vocab.get_idx("EOS"))
		return encode_words


	def build_dialog_pair(self):
		for item in self.dialog.dial_data:
			item_len=len(item)
			if item_len%2 !=0:
				item=item[0:int(item_len/2)*2]
			for index in range(0,len(item),2):
				self.dialog_pair.append(list((item[index],item[index+1])))

	def encode_dialog_pair(self):
		for dialog_pair in self.dialog_pair:
			encoded_pair=[self.encode_sent(self.dialog.utterance.get_sent(utter_idx)) for utter_idx in dialog_pair]
			if len(encoded_pair[0])>1 and len(encoded_pair[0])<max_len \
					and len(encoded_pair[1])>1 and len(encoded_pair[1])<max_len:
				self.encoded_dial_pair.append(encoded_pair)
#		print('encoded_dial_pair:{}'.format(self.encoded_dial_pair))


	def save_dialog_pair(self,path):
		with open(path,'w') as f:
			json.dump(self.encoded_dial_pair,f)

class DataLoader:
	def __init__(self,batch_size,dial_path,vocab_path):
		self.batch_size=batch_size
		self.vocab=Vocabulary()
		self.vocab.load_vocab(vocab_path)
		with open(dial_path,'r') as f:
			self.data=json.load(f)
#		self.data=sorted(self.data,key=lambda item:len(item[0]),reverse=True)
		
	def seq_pad(self,seq,max_len):
		seq+=[self.vocab.get_idx('PADED') for i in range(max_len-len(seq))]
		return seq



	def get_batch(self):
		source=list()
		target=list()


		batch_num=int(len(self.data)/self.batch_size)
		batch_list=[self.data[k:k+self.batch_size] for k in range(0,batch_num*self.batch_size,self.batch_size)]
		for batch in batch_list:
			batch=sorted(batch,key=lambda item:len(item[0]),reverse=True)
			source=[item[0] for item in batch]
			target=[item[1] for item in batch]

			sour_len=[len(item) for item in source]
			sour_max=max(sour_len)
#			print(sour_len)
#			print(source)

			targ_len=[len(item) for item in target]
			targ_max=max(targ_len)

			source=[self.seq_pad(item,sour_max) for item in source]
			target=[self.seq_pad(item,targ_max) for item in target]

			inp=Variable(torch.LongTensor(source)).transpose(0,1)
			oup=Variable(torch.LongTensor(target)).transpose(0,1)
			yield (inp,sour_len),(oup,targ_len)
				

if __name__=='__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('--corpus_path',
			type=str,
			default='../data/movie_dialogue/',
			help='corpus data path')
	parser.add_argument('--dialog_path',
			type=str,
			default='../data/dialog_data.json',
			help='generated dailog data path')
	parser.add_argument('--vocab_path',
			type=str,
			default='../data/vocabulary.json',
			help='generated vocabulary path')
	parser.add_argument('--vocab_threshold',
			type=int,
			default=10)
	parser.add_argument('--dialog_pair_path',
			type=str,
			default='../data/dialog_pair.json')
	parser.add_argument('--is_freq',
			type=bool,
			default=True,
			help='generate vocabulary by frequency')


	args=parser.parse_args()
	
	
	dialpair=DialPair(args.corpus_path,args.dialog_path,args.vocab_path,args.vocab_threshold,
						args.is_freq)
	dialpair.build_dialog_pair()
	dialpair.encode_dialog_pair()
	dialpair.save_dialog_pair(args.dialog_pair_path)
#	for index,dialog in enumerate(dataloader.dialog_pair):
#		print('index:{}\n utter 1:{}\n utter 2:{}'.format(index,dialog[0],dialog[1]))
	dataloader=DataLoader(5,args.dialog_pair_path,args.vocab_path)

	batch_generator=dataloader.get_batch()
	for source,target in batch_generator:
		sour_len=sorted(source[1])
		targ_len=sorted(target[1])
		print('source:{}\ntaget:{}'.format(source[0],target[0])) 
		if sour_len[0]<=0 or targ_len[0]<=0:
			print('sour_len:{}   targ_len:{}'.format(sour_len,targ_len))
			break

	print('conversation_num:{}'.format(len(dataloader.data)))
#		print('source:{}\ntaget:{}'.format(source[0],target[0]))	

	
#	dial=Conversation()
#	dial.load_dial_data(args.dial_path)
#	with open('dial_utter.json','w') as f:
#		json.dump(dial.utterance.index2sent,f)

#	with open('dial_info.json','w') as f:
#		json.dump(dial.dial_data,f)
#	print(dial.utterance.index2sent)

#	with open('idx2word.json','w') as f:
#		json.dump(dial.vocab.idx2word,f)
