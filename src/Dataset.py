import torch
import json
import logging
import os

from vocabulary import Vocabulary
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader


logging.basicConfig(level=logging.DEBUG,format="%(asctime)s -- %(message)s")
logger=logging.getLogger(__name__)

class DataSet(Dataset):

    def __init__(self,data_path,vocab_path):
        super(Dataset, self).__init__()
        with open(data_path,'r') as f:
            self.data=json.load(f)

        with open(vocab_path,'r') as f:
            self.vocab=Vocabulary()
            self.vocab.load_vocab(vocab_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dial_collate_fn(batch):
    batch = sorted(batch,key=lambda dial_pair: len(dial_pair[0]), reverse=True)
    sour_leng = [len(dial_pair[0]) for dial_pair in batch]
    targ_leng = [len(dial_pair[1]) for dial_pair in batch]

    sour_seq = [dial_pair[0] for dial_pair in batch]
    targ_seq = [dial_pair[1] for dial_pair in batch]

    sour_seq = Variable(torch.LongTensor([Vocabulary.seq_pad(sent, max(sour_leng)) for sent in sour_seq]))
    targ_seq = Variable(torch.LongTensor([Vocabulary.seq_pad(sent, max(targ_leng)) for sent in targ_seq]))
    sour_seq = sour_seq.transpose(0, 1)
    targ_seq = targ_seq.transpose(0, 1)

    return (sour_seq, sour_leng, targ_seq, targ_leng)



def get_dataloader(data_path,vocab_path,batch_size):
    dataset=DataSet(data_path,vocab_path)
    print(len(dataset))
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=1,
                      drop_last=True,
                      collate_fn=dial_collate_fn)

if __name__=='__main__':
    data_dir = '../experiment_data/data'
    data_path = os.path.join(data_dir, 'train_auto.json')
    vocab_path = os.path.join(data_dir, 'vocabulary.json')
    dataloader = get_dataloader(data_path, vocab_path, 5)
    for idx,batch in enumerate(dataloader):
        print(batch)
