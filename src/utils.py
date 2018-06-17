
# coding: utf-8

# In[ ]:


import torch
import os
import logging

import torch.nn as nn

from checkpoint import Checkpoint
from torch.autograd import Variable
from Dataset import get_dataloader
from Seq2Seq_model import Seq2Seq
from tensorboard_logger import log_value,configure
from vocabulary import Vocabulary

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s -- %(message)s")
logger=logging.getLogger(__name__)



def check_gradient(model):
    for idx,module in enumerate(model.tran_layer):
        for sub_idx,sub_module in enumerate(module):
            if sub_idx==1 and idx==2:
 #               logger.info(sub_module)
                logger.info('gradient of {} module is :\n{}'.format(sub_idx,sub_module.weight.grad.data[0]))
                return


def save_test_sent(data_dir,file_name,sent_list):
    data_file=os.path.join(data_dir,file_name)
    with open(data_file,'w') as f:
        for item in sent_list:
            f.write('source_text: '+item[0]+'\n')
            f.write('target_text: '+ item[1]+'\n')
            for pred_sent in item[2]:
                f.write('predict_text: '+pred_sent[0]+'   prob:'+str(pred_sent[1])+'\n')
            f.write('\n')

def process_sent_list(vocab,sour_seq,targ_seq,sent_list):
    sour_sent_list = [vocab.decode_sent(sent) for sent in sour_seq.transpose(0, 1).data.tolist()]
    targ_sent_list = [vocab.decode_sent(sent) for sent in targ_seq.transpose(0,1).data.tolist()]
    pred_sent_list = [(vocab.decode_sent(sent[0]), sent[1]) for sent in sent_list]
#    logger.info('sent_list:{}\n predict:{}'.format(sent_list,pred_sent_list))
    test_sent_pair_list = [(sour_sent_list[idx],targ_sent_list[idx], pred_sent_list) for idx in range(len(sour_sent_list))]
    logger.info('test_sent_pair_list:{}'.format(test_sent_pair_list))

    return test_sent_pair_list


def train_batch(model,optimizer,criterion,batch):

    sour,sour_len,targ,targ_len=batch
    hidd_state=model.encoder_forward(sour,sour_len)
    out_prob=model.decoder_forward(targ,targ_len,hidd_state,1)
    out_prob=out_prob.view(-1,out_prob.size(2))
    targ=targ.view(-1)
    optimizer.zero_grad()
    loss=criterion(out_prob,targ)
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(),5)
    optimizer.step()

    return loss.item()

def validate_batch(model,criterion,batch):
    sour,sour_len,targ,targ_len=batch
    hidd_state=model.encoder_forward(sour,sour_len)
    out_prob=model.decoder_forward(targ,targ_len,hidd_state,0)
    out_prob=out_prob.view(-1,out_prob.size(2))
    targ=targ.view(-1)
    loss = criterion(out_prob, targ)

    return loss.item()


    
def train(args):

    configure(args['log_dir'])

    dial_data=get_dataloader(os.path.join(args['data_dir'],'encoded_train_dialogue_pair.json'),
                             os.path.join(args['data_dir'],'vocabulary.json'),
                             args['batch_size'])
    vocab=Vocabulary()
    vocab.load_vocab(os.path.join(args['data_dir'],'vocabulary.json'))
    args['voca_size']=len(vocab.word2idx)

    model=Seq2Seq(args).cuda() if torch.cuda.is_available() else Seq2Seq(args)

    optimizer=torch.optim.Adam(model.parameters(),lr=args['lr'])

    criterion=nn.NLLLoss(ignore_index=vocab.get_idx('PADED'))

    min_valid_loss=float('inf')

    for epoch in range(args['epoches']):
        for batch_idx,(sour,sour_len,targ,targ_len) in enumerate(dial_data):
            if torch.cuda.is_available():
                sour=sour.cuda()
                targ=targ.cuda()
            loss=train_batch(model,optimizer,criterion,(sour,sour_len,targ,targ_len))

            logger.info('training loss:{}'.format(loss))
            log_value('CrossEntropy loss', loss, epoch*len(dial_data)+batch_idx)

            if  (batch_idx+epoch*len(dial_data))% args['valid_step']==0 :
                valid_loader=get_dataloader(os.path.join(args['data_dir'],'encoded_valid_dialogue_pair.json'),
                                            os.path.join(args['data_dir'],'vocabulary.json'),
                                            args['batch_size'])
                valid_loss = validate(model,valid_loader,criterion)

                log_value('valid loss', valid_loss,
                          int((batch_idx+epoch*len(dial_data))/ args['valid_step']))
                logger.info('valid_step:{} valid_loss:{}'.format(
                    int((batch_idx+epoch*len(dial_data))/ args['valid_step']), valid_loss))

                checkpoint=Checkpoint(model,optimizer,epoch,batch_idx)
                checkpoint.save(args['exp_dir'])

        
def validate(model,dial_data,criterion):
    total_loss=0

    for batch_idx,(sour,sour_len,targ,targ_len) in enumerate(dial_data):
        if torch.cuda.is_available():
            sour=sour.cuda()
            targ=targ.cuda()

        loss=validate_batch(model,criterion,(sour,sour_len,targ,targ_len))

        total_loss+=loss

    return total_loss/len(dial_data)


def test(args):

    vocab=Vocabulary()
    vocab.load_vocab(os.path.join(args['data_dir'],'vocabulary.json'))
    args['voca_size']=vocab.get_vocab_size()
    test_data=get_dataloader(os.path.join(args['data_dir'],'encoded_test_dialogue_pair.json'),
                             os.path.join(args['data_dir'],'vocabulary.json'),1)
    test_sent_pair_list=[]

    model=Seq2Seq(args).eval()
    if torch.cuda.is_available():
        model=model.cuda()

    path=Checkpoint.get_latest_checkpoint(args['exp_dir'])
    model.load_state_dict(torch.load(os.path.join(path,'model.pt')))

    for batch_idx,(sour,sour_len,targ,targ_len) in enumerate(test_data):
        if torch.cuda.is_available():
            sour=sour.cuda()
            targ=targ.cuda()
        enco_hidd_state=model.encoder_forward(sour,sour_len)
        out_prob = model.decoder_forward(targ,targ_len,enco_hidd_state,0)
        sent_list = [(out_prob.topk(1)[1].view(-1).tolist(), 0)]
        test_sent_pair_list+=process_sent_list(vocab,sour,targ,sent_list)
#   logger.info('batch_idx:{} \nsent:{}'.format(batch_idx,test_sent_pair_list))

    save_test_sent(args['exp_data'],'generated_test_sent.txt',test_sent_pair_list)


