# coding: utf-8

# In[7]:


# coding: utf-8

# In[7]:


import torch
import random
import logging

import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from torch.autograd import Variable


class Seq2Seq(nn.Module):

    def __init__(self, args):
        # args is a dictionary which stores all paramteres setting RNN Encoder
        super(Seq2Seq, self).__init__()
        self.hidd_dime = args['hidd_dime']
        self.word_dime = args['word_dime']
        self.hidd_laye = args['hidd_layer']
        self.voca_size = args['voca_size']

        self.sos_idx = 0
        self.eos_idx = 1
        self.pad_idx = 2

        self.drop_out = nn.Dropout(args['drop_rate'])

        if args['RNN_type'] == 'GRU':
            rnn = nn.GRU
            self.rnn_type = 0

        elif args['RNN_type'] == 'LSTM':
            rnn = nn.LSTM
            self.rnn_type = 1

        else:
            logger.error(args['RNN_type']+' illegle RNN type')
            return

        self.hidd_fact = self.hidd_laye * (2 if args['bidirectional'] else 1)

        self.embd_matrix = nn.Embedding(self.voca_size, self.word_dime)

        self.enco_rnn = rnn(self.word_dime, self.hidd_dime, self.hidd_laye, bidirectional=args['bidirectional'])

        self.embd_matrix2=nn.Embedding(self.voca_size, self.word_dime)

        self.deco_rnn = rnn(self.word_dime, self.hidd_dime, self.hidd_laye, bidirectional=args['bidirectional'])
        self.hidd_to_voca = nn.Linear(self.hidd_dime * int(self.hidd_fact / self.hidd_laye), self.voca_size)

        self.log_func = nn.LogSoftmax()

    def encoder_forward(self, sour_seq, seq_len):
        # sour_seq is a variable storing a batch of sequence
        # seq_len is a list storing length of each sequence

        init_range = 0.01

        hidd_state = Variable(torch.zeros(self.hidd_fact, sour_seq.size(1), self.hidd_dime))
        hidd_state = torch.nn.init.uniform(hidd_state, -init_range, init_range)

        if self.rnn_type == 1:
            cell_state = Variable(torch.zeros(hidd_state.shape))
            cell_state = torch.nn.init.uniform(cell_state, -init_range, init_range)
            hidd_state = (hidd_state, cell_state)

        if torch.cuda.is_available():
            if type(hidd_state) is tuple:
                hidd_state = (hidd_state.cuda(), cell_state.cuda())
            else:
                hidd_state = hidd_state.cuda()

        embd_seq = self.embd_matrix(sour_seq)
        pack_seq = rnn_utils.pack_padded_sequence(self.drop_out(embd_seq), seq_len)
        out, hidd_state = self.enco_rnn(pack_seq, hidd_state)

        return hidd_state

    def encoder_forward_by_sentence(self, seq_list,seq_len):

        init_range = 0.01
        hidd_state_list=[]

        tranf_seq_list=seq_list.transpose(0,1)
        batch_size=tranf_seq_list.size(0)

        for idx in range(batch_size):
            sent_seq=Variable(tranf_seq_list[idx]).view(-1,1)

            sent_hidd_state=Variable(torch.zeros(self.hidd_fact,1,self.hidd_dime))
            sent_hidd_state=torch.nn.init.uniform(sent_hidd_state,-init_range,init_range)

            if self.rnn_type == 1:
                sent_cell_state = Variable(torch.zeros(sent_hidd_state.shape))
                sent_cell_state = torch.nn.init.uniform(sent_cell_state, -init_range, init_range)
                sent_hidd_state = (sent_hidd_state, sent_cell_state)

            if torch.cuda.is_available():
                if type(sent_hidd_state) is tuple:
                    sent_hidd_state = (sent_hidd_state.cuda(), sent_cell_state.cuda())
                else:
                    sent_hidd_state = sent_hidd_state.cuda()

            embd_seq = self.embd_matrix(sent_seq)
            pack_seq = rnn_utils.pack_padded_sequence(self.drop_out(embd_seq), [seq_len[idx]])
            out, sent_hidd_state = self.enco_rnn(pack_seq, sent_hidd_state)
            hidd_state_list.append(sent_hidd_state)

        hidd_state=torch.stack(hidd_state_list)

        return hidd_state




    def decoder_forward_step(self, word_list, hidd_state):
        embd_word = self.embd_matrix2(word_list)  # embd_word: 1xBatchx Word_dim
        out, hidd_state = self.deco_rnn(embd_word, hidd_state)
        out = self.hidd_to_voca(out.squeeze(0))  # out: Batch x Vocab
        out = self.log_func(out)

        return out, hidd_state

    def decoder_forward(self, seq_list, seq_len, hidd_state, teac_rate):
        # teac_rate indicates how many target words will be inputted to decoder

        max_len = max(seq_len)

        out_matrix = Variable(torch.zeros(max_len, seq_list.size(1), self.voca_size))
        out_matrix = out_matrix.cuda() if torch.cuda.is_available() else out_matrix

        word_list = Variable(torch.LongTensor([self.sos_idx] * seq_list.size(1))).view(1, -1)
        for idx in range(0, max_len):
            word_list = word_list.cuda() if torch.cuda.is_available() else word_list
            out_step, hidd_state = self.decoder_forward_step(word_list, hidd_state)
            # out_step: Batch x Word_dim
            if random.random() >= teac_rate:
                word_list = out_step.topk(1)[1].view(1, -1).contiguous()

            else:
                word_list = seq_list[idx].contiguous().view(1, -1)

            out_matrix[idx] = out_step
        return out_matrix.contiguous()

    def decoder_with_beam_search(self, context_vector):
        max_len = 20
        beam_size = 10
        dead_num = 0
        hidd_factor = context_vector.size(0)

        generated_sents = []

        hidden_vector = context_vector.repeat(1, beam_size, 1)
        inp = Variable(torch.LongTensor([1] * beam_size)).view(1, beam_size)

        if torch.cuda.is_available():
            inp = inp.cuda()
            hidden_vector = hidden_vector.cuda()

        for step in range(max_len):

            out_step, hidden_vector = self.decoder_forward_step(inp, hidden_vector)
            word_prob = out_step.topk(beam_size)[0]
            word_idx = out_step.topk(beam_size)[1]

            tmp_words = []

            if step == 0:
                for idx in range(beam_size):
                    word = []
                    word.append([word_idx.data[0][idx].tolist()])
                    word.append([word_prob.data[0][idx].tolist()])
                    word.append([hidden_vector.data[0][idx]])

                    tmp_words.append(word)


            else:
                for idx in range(beam_size - dead_num):
                    for sub_idx in range(beam_size):
                        word = []
                        word.append(candidate_words[idx][0] + [word_idx.data[idx][sub_idx].tolist()])
                        word.append([candidate_words[idx][1][0] +
                                     word_prob.data[idx][sub_idx].tolist()])
                        word.append([hidden_vector.data[0][idx]])

                        tmp_words.append(word)

            # print(step)

            tmp_words = sorted(tmp_words, key=lambda x: x[1][0] / len(x[0]), reverse=True)

            tmp_words = tmp_words[0:beam_size - dead_num]

            # if step==5:
            # print(tmp_words)

            candidate_words = []

            # print(tmp_words)

            for idx in range(len(tmp_words)):
                if tmp_words[idx][0][-1] == self.eos_idx:
                    generated_sents.append((tmp_words[idx][0], tmp_words[idx][1][0]))
                    dead_num += 1
                    # print('generted words')

                else:
                    candidate_words.append(tmp_words[idx])

            # print(candidate_words)

            if beam_size - dead_num == 0:
                return generated_sents

            hidden_vector = Variable(torch.zeros(hidd_factor, beam_size - dead_num, hidden_vector.size(2)))
            inp = Variable(torch.LongTensor([0] * (beam_size - dead_num)))

            for idx in range(beam_size - dead_num):
                hidden_vector.data[0][idx] = candidate_words[idx][2][0]
                inp.data[idx] = candidate_words[idx][0][-1]

            if torch.cuda.is_available():
                hidden_vector = hidden_vector.cuda()
                inp = inp.cuda()

            hidden_vector = hidden_vector.view(hidd_factor, beam_size - dead_num, -1)
            inp = inp.view(1, beam_size - dead_num)

            # if step==1:
            # print(candidate_words)
            # print(inp)

        if step == max_len - 1:
            for idx in range(len(candidate_words)):
                generated_sents.append((candidate_words[idx][0], candidate_words[idx][1][0]))

        #        print(generated_sents)

        return generated_sents


logging.basicConfig(format='%(asctime)-15s %(message)s', level='DEBUG')
logger = logging.getLogger(__name__)

# In[8]:


