import re
import json
import os
import sys
import nltk
import random

from vocab import Vocabulary

max_len=30
min_len=1

class Conversation:

	def __init__(self):
		self.raw_utter=[]
		self.raw_conv={}
		self.train_conv={}
		self.validate_conv={}
		self.test_conv={}
		self.encoded_train_conv={}
		self.encoded_test_conv={}
		self.encoded_validate_conv={}

	def normalizeString(self,s):
		s = s.lower().strip()
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?\']+", r" ", s)
		return s

	def split_conv(self,train_ratio=0.8,validate_ratio=0.1,test_ratio=0.1):

		train_size=int(len(self.raw_conv)*train_ratio)
		validate_size=int(len(self.raw_conv)*validate_ratio)
		test_size=len(self.raw_conv)-train_size-validate_size

		while len(self.validate_conv)<validate_size:
			index=random.randint(0,len(self.raw_conv)-1)
			if index not in self.validate_conv:
				self.validate_conv[index]=self.raw_conv[index]
			

		while len(self.test_conv)<test_size:
			index=random.randint(0,len(self.raw_conv)-1)
			if index not in self.test_conv and index not in self.validate_conv:
				self.test_conv[index]=self.raw_conv[index]

		for index in range(len(self.raw_conv)):
			if index not in self.validate_conv and index not in self.test_conv:
				self.train_conv[index]=self.raw_conv[index]

		
	def get_raw_utter(self, source_path):
		with open(source_path) as f:
			for utter in f:
				utter=self.normalizeString(utter)
				words=nltk.word_tokenize(utter)
				if len(words)>min_len and len(words)<max_len:
					self.raw_utter.append(words) 

	def get_raw_conv(self,source_path):
		self.get_raw_utter(source_path)

		for index in range(len(self.raw_utter)-1):
			self.raw_conv[index]=[]
			self.raw_conv[index].append(self.raw_utter[index])
			self.raw_conv[index].append(self.raw_utter[index+1])

		self.split_conv()


	def save_conv(self,dst_path):
		raw_data_path=os.path.join(dst_path,'raw_data.json')

		with open(raw_data_path,'w') as f:
			json.dump({
						'raw_utter':self.raw_utter,
						'raw_conv':self.raw_conv,
						'train_conv':self.train_conv,
						'test_conv':self.test_conv,
						'validate_conv':self.validate_conv},f)

		train_conv_path=os.path.join(dst_path,'encoded_train_conv.json')
		test_conv_path=os.path.join(dst_path,'encoded_test_conv.json')
		validate_conv_path=os.path.join(dst_path,'encoded_validate_conv.json')

		with open(train_conv_path,'w') as f:
			json.dump({'encoded_train_conv':self.encoded_train_conv},f)

		with open(test_conv_path,'w') as f:
			json.dump({'encoded_test_conv':self.encoded_test_conv},f)

		with open(validate_conv_path,'w') as f:
			json.dump({'encoded_validate_conv':self.encoded_validate_conv},f)


	def load_data(self,dst_path):

		with open(dst_path,'r') as f:
			data=json.load(f)

		return data


	def load_conv(self,dst_path):

		raw_data_path=os.path.join(dst_path,'raw_data.json')

		raw_data=self.load_data(raw_data_path)

		self.raw_utter=raw_data['raw_utter']
		self.raw_conv=raw_data['raw_conv']
		self.train_conv=raw_data['train_conv']
		self.test_conv=raw_data['test_conv']
		self.validate_conv=raw_data['validate_conv']

		train_conv_path=os.path.join(dst_path,'encoded_train_conv.json')
		test_conv_path=os.path.join(dst_path,'encoded_test_conv.json')
		validate_conv_path=os.path.join(dst_path,'encoded_validate_conv.json')

		self.encoded_train_conv=self.load_data(train_conv_path)

		self.encoded_test_conv=self.load_data(test_conv_path)

		self.encoded_validate_conv=self.load_data(validate_conv_path)




	def get_encoded_conv(self,dst_path,vocab_path):
#		self.get_raw_conv(source_path)
		voc=Vocabulary()
		voc.load_vocab(vocab_path)
		for train_item in self.train_conv.items():
			self.encoded_train_conv[train_item[0]]=[]
			for sent in train_item[1]:
				self.encoded_train_conv[train_item[0]].append(voc.encode_sent(sent))

		for valid_item in self.validate_conv.items():
			self.encoded_validate_conv[valid_item[0]]=[]
			for sent in valid_item[1]:
				self.encoded_validate_conv[valid_item[0]].append(voc.encode_sent(sent))

		for test_item in self.test_conv.items():
			self.encoded_test_conv[test_item[0]]=[]
			for sent in test_item[1]:
				self.encoded_test_conv[test_item[0]].append(voc.encode_sent(sent))

		
		self.save_conv(dst_path)

