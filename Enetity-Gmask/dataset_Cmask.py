# coding: utf-8
# author: noctli
import json
import pickle
import logging
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain
import random
import copy
from nltk.chunk import ne_chunk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import sent_tokenize
# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>", "<mask>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>", 'mask_token': "<mask>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]


def tokenize(obj,tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)
def nltk_enetiy(questions_nltk,answers_nltk):
    ques_nltk = []
    ans_nltk = []
    filter_word = ["heard","video","didnt","clip","i","end"]
    for i in range(0,len(questions_nltk)):
        sentences = sent_tokenize(questions_nltk[i])
        for sentence in sentences:
            words = word_tokenize(sentence)
            ques_arr = [0]*len(words)
            tagged = pos_tag(words)
            chunks = ne_chunk(tagged)
            for index,chunk in enumerate(tagged):
                if(chunk[1].startswith('NN')):
                    a = chunk[0]
                    if(a in filter_word or 'thing' in a):
                        continue
                    else:
                        ques_arr[index] = 1
            ques_nltk.append(ques_arr)
        sentences = sent_tokenize(answers_nltk[i])
        for sentence in sentences:
            words = word_tokenize(sentence)
            ans_arr = [0] * len(words)
            tagged = pos_tag(words)
            chunks = ne_chunk(tagged)
            for index, chunk in enumerate(tagged):
                if (chunk[1].startswith('NN')):
                    a = chunk[0]
                    if (a in filter_word or 'thing' in a):
                        continue
                    else:
                        ans_arr[index] = 1
            ans_nltk.append(ans_arr)
    return ques_nltk,ans_nltk
def get_dataset(tokenizer, data_file, feature_path=None, undisclosed_only=False, n_history=3):
    
    dialog_data = json.load(open(data_file, 'r'))
    dialog_list = []
    vid_set = set()
    for dialog in dialog_data['dialogs']:
        caption = [tokenize(dialog['caption'],tokenizer)] + [tokenize(dialog['summary'],tokenizer)]
        questions = [tokenize(d['question'],tokenizer) for d in dialog['dialog']]
        questions_nltk = [copy.deepcopy(d['question']) for d in dialog['dialog']]
        answers = [tokenize(d['answer'],tokenizer) for d in dialog['dialog']]
        answers_nltk = [copy.deepcopy(d['answer']) for d in dialog['dialog']]
        questions_enetiy,answers_enetiy = nltk_enetiy(questions_nltk,answers_nltk)
        vid = dialog["image_id"]
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions) - 1, len(questions))
        else:
            it = range(len(questions))
        qalist=[]
        qalist_nltk=[]
        history = []
        history_nltk=[]
        if undisclosed_only:
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            question_nltk = questions_enetiy[n]
            answer = answers[n]
            answer_nltk = answers_enetiy[n]
            history.append(question)
            history_nltk.append(question_nltk)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'history_nltk': [question_nltk], 'answer': answer, 'caption': caption}
            else:
                item = {'vid': vid, 'history': history,'history_nltk': history_nltk, 'answer': answer, 'caption': caption}
            dialog_list.append(item)
            qalist.append(question)
            qalist_nltk.append(question_nltk)
            qalist.append(answer)
            qalist_nltk.append(answer_nltk)
            history=qalist[max(-len(qalist),-n_history*2):]
            history_nltk=qalist_nltk[max(-len(qalist_nltk),-n_history*2):]

    all_features = {}
    if feature_path is not None:
        fea_types = ['vggish', 'i3d_flow', 'i3d_rgb']
        dataname = '<FeaType>/<ImageID>.npy'
        for ftype in fea_types:
            if undisclosed_only:
                basename = dataname.replace('<FeaType>', ftype+'_testset')
            else:
                basename = dataname.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filename = basename.replace('<ImageID>', vid)
                filepath = feature_path + filename
                features[vid] = (filepath, filepath)
            all_features[ftype] = features
        return dialog_list, all_features

    return dialog_list


class AVSDDataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0.5, train=True):
        self.dialogs = dialogs
        self.features = features
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid']
        his = self.dialogs[index]['history']
        cap = self.dialogs[index]['caption']
        ans = self.dialogs[index]['answer']
        his_nltk = self.dialogs[index]['history_nltk']
        if np.random.rand() < self.drop_rate:
            instance, _ = build_input_from_segments(his_nltk, cap, his, ans, self.tokenizer, video=False, drop_caption=True, train=self.train)
        else:
            instance, _ =build_input_from_segments(his_nltk, cap, his, ans, self.tokenizer, video=False, drop_caption=False, train=self.train)
        input_ids = torch.Tensor(instance["input_ids"]).long()
        token_type_ids = torch.Tensor(instance["token_type_ids"]).long()
        lm_labels = torch.Tensor(instance["lm_labels"]).long()
        input_nltk = torch.Tensor(instance["sequence_nltk"]).long()
        if self.features is not None:
            try:
                vgg = np.load(self.features[0]["vggish"][vid][0])
                i3d_flow = np.load(self.features[0]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[0]["i3d_rgb"][vid][0])
            except KeyError:
                vgg = np.load(self.features[1]["vggish"][vid][0])
                i3d_flow = np.load(self.features[1]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[1]["i3d_rgb"][vid][0])

            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], 1)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], 1)]

            vgg = torch.from_numpy(vgg).float()
            i3d_flow = torch.from_numpy(sample_i3d_flow).float()
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float()
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1)

            return input_ids, token_type_ids, lm_labels, i3d, input_nltk
        else:
            return input_ids, token_type_ids, lm_labels
# 对抽象名词进行mask
def choice_mask_nltk(input_ids_list,nltk_list, lm_labels_list, mask_token):
    for i,nltk_v in enumerate(nltk_list):
        nltk_index = [index for index,item in enumerate(nltk_v) if item==1]
        for j in nltk_index:
            lm_labels_list[i][j] = input_ids_list[i][j]
            input_ids_list[i][j] = mask_token
    return input_ids_list,lm_labels_list
def cross_Mask(input_ids_list,token_type_ids_list,i3d_list,mask_token):
    input_ids_list1, token_type_ids_list1, i3d_list1 = [],[],[]
    i3d,history,question,answer = i3d_list,[],[],[]
    history_ids, question_ids, answer_ids = [], [], []
    speaker1,speaker2 ,eos = 50259,50260 , 50258
    speaker1 = torch.tensor(speaker1)
    speaker2 = torch.tensor(speaker2)
    eos = torch.tensor(eos)
    for i in range(0,len(input_ids_list)):
        answer_start = 0
        question_start = 0
        answer_flag = True
        question_flag = True
        for j in range(0,len(token_type_ids_list[i])):
            if(token_type_ids_list[i][-1*(j+1)] == speaker1 and answer_flag):
                answer_start = -1*(j+1) + 1
                answer_flag = False
                continue
            if(answer_flag):
                continue
            else:
                if(token_type_ids_list[i][-1*(j+1)] == speaker2 and question_flag):
                    question_start = -1*(j+1) + 1
                    question_flag = False
                    continue
        history.append(input_ids_list[i][0:question_start])
        question.append(input_ids_list[i][question_start:answer_start])
        answer.append(input_ids_list[i][answer_start:])
    for i in range(0,len(input_ids_list)):
        if np.random.rand() < 0.25:
            i3d_list_ = torch.ones((i3d_list[i].size(0), i3d_list[i].size(1))).long() * mask_token
            i3d_list1.append(i3d_list_)
            input_ids_list1.append(input_ids_list[i])
            continue
        elif np.random.rand() < 0.5:
            i3d_list1.append(i3d_list[i])
            a = history[i]
            a[(a != speaker1) & (a != speaker2)] = mask_token
            input_ids_list1.append(torch.cat([a, question[i], answer[i]], dim=0))
            continue
        elif np.random.rand() < 0.75:
            i3d_list1.append(i3d_list[i])
            a = question[i]
            a[(a != speaker1) & (a != speaker2)] = mask_token
            input_ids_list1.append(torch.cat([history[i], a, answer[i]], dim=0))
            continue
        else:
            i3d_list1.append(i3d_list[i])
            a = answer[i]
            a[(a != speaker1) & (a != speaker2) & (a != eos)] = mask_token
            input_ids_list1.append(torch.cat([history[i], question[i], a], dim=0))
            continue
    return input_ids_list1 , i3d_list1
def cross_Contrastive(input_ids_list,token_type_ids_list,i3d_list,lm_labels_list):
    if len(input_ids_list) == 1:
        return input_ids_list,token_type_ids_list,i3d_list,lm_labels_list
    input_ids_list1, token_type_ids_list1, i3d_list1,lm_labels_list1 = [],[],[],[]
    i3d,history,question,answer = i3d_list,[],[],[]
    history_ids, question_ids, answer_ids = [], [], []
    history_labels, question_labels, answer_labels = [], [], []
    speaker1,speaker2 = 50259,50260
    speaker1 = torch.tensor(speaker1)
    speaker2 = torch.tensor(speaker2)
    for i in range(0,len(input_ids_list)):
        answer_start = 0
        question_start = 0
        answer_flag = True
        question_flag = True
        for j in range(0,len(token_type_ids_list[i])):
            if(token_type_ids_list[i][-1*(j+1)] == speaker1 and answer_flag):
                answer_start = -1*(j+1) + 1
                answer_flag = False
                continue
            if(answer_flag):
                continue
            else:
                if(token_type_ids_list[i][-1*(j+1)] == speaker2 and question_flag):
                    question_start = -1*(j+1) + 1
                    question_flag = False
                    continue
        history.append(input_ids_list[i][0:question_start])
        question.append(input_ids_list[i][question_start:answer_start])
        answer.append(input_ids_list[i][answer_start:])

        history_ids.append(token_type_ids_list[i][0:question_start])
        question_ids.append(token_type_ids_list[i][question_start:answer_start])
        answer_ids.append(token_type_ids_list[i][answer_start:])

        history_labels.append(lm_labels_list[i][0:question_start])
        question_labels.append(lm_labels_list[i][question_start:answer_start])
        answer_labels.append(lm_labels_list[i][answer_start:])
    for i in range(0,len(input_ids_list)):
        all_index = list(range(0,len(input_ids_list)))
        all_index.remove(i)
        option_ind = random.choice(all_index)
        if np.random.rand() < 0.33:
            i3d_list1.append(i3d_list[option_ind])
            input_ids_list1.append(input_ids_list[i])
            token_type_ids_list1.append(token_type_ids_list[i])
            lm_labels_list1.append(lm_labels_list[i])
            continue
        elif np.random.rand() < 0.66:
            i3d_list1.append(i3d_list[i])
            input_ids_list1.append(torch.cat([history[option_ind], question[i], answer[i]], dim=0))
            token_type_ids_list1.append(torch.cat([history_ids[option_ind], question_ids[i], answer_ids[i]], dim=0))
            lm_labels_list1.append(torch.cat([history_labels[option_ind], question_labels[i], answer_labels[i]], dim=0))
            continue
        else :
            i3d_list1.append(i3d_list[i])
            input_ids_list1.append(torch.cat([history[i], question[option_ind], answer[i]], dim=0))
            token_type_ids_list1.append(torch.cat([history_ids[i], question_ids[option_ind], answer_ids[i]], dim=0))
            lm_labels_list1.append(torch.cat([history_labels[i], question_labels[option_ind], answer_labels[i]], dim=0))
    return input_ids_list1, token_type_ids_list1, i3d_list1,lm_labels_list1

def choice_mask(input_ids_list, token_type_ids_list, lm_labels_list, mask_token):

    speaker1, speaker2, eos, cap, bos = 50259, 50260, 50258, 50262, 50257
    speaker1 = torch.tensor(speaker1)
    speaker2 = torch.tensor(speaker2)
    eos = torch.tensor(eos)
    cap = torch.tensor(cap)
    bos = torch.tensor(bos)
    for i in range(0, len(input_ids_list)):
        cur_index = []
        answer_start = 0
        for j in range(0, len(token_type_ids_list[i])):
            if (token_type_ids_list[i][-1 * (j + 1)] == speaker1):
                answer_start = -1 * (j + 1) + 1
                break
        for index,value in enumerate(input_ids_list[i][:answer_start]):
            if(value != speaker1 and value != speaker2 and value != eos and value != cap and value != bos):
                cur_index.append(index)
        random.shuffle(cur_index)
        end_index = int(len(cur_index)*0.3)
        for index1 in cur_index[:end_index]:
            lm_labels_list[i][index1] = input_ids_list[i][index1]
            input_ids_list[i][index1] = mask_token
    return input_ids_list, lm_labels_list

def collate_fn(batch, pad_token ,mask_token , features=None):
    def padding(seq, pad_token):# pad_token:50263
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    input_ids_list, token_type_ids_list, lm_labels_list, i3d_list, nltk_list = [], [], [], [], []
    for i in batch:
        input_ids_list.append(i[0])
        token_type_ids_list.append(i[1])
        lm_labels_list.append(i[2])
        nltk_list.append(i[4])
        if features is not None:
            i3d_list.append(i[3])
    if np.random.rand() < 0.5:
        input_ids_list, token_type_ids_list, i3d_list,lm_labels_list = cross_Contrastive(input_ids_list,token_type_ids_list,i3d_list,lm_labels_list)
        #input_ids_list, i3d_list = cross_Mask(input_ids_list,token_type_ids_list,i3d_list,mask_token)
    else :
        input_ids_list,lm_labels_list = choice_mask_nltk(input_ids_list,nltk_list, lm_labels_list, mask_token)
    input_ids = padding(input_ids_list, pad_token)
    token_type_ids = padding(token_type_ids_list, pad_token)
    lm_labels = padding(lm_labels_list, -1)
    input_mask = input_ids != pad_token
    if features is not None:
        i3d = padding(i3d_list, pad_token)
        i3d_mask = torch.sum(i3d != 1, dim=2) != 0
        input_mask = torch.cat([i3d_mask, input_mask], dim=1)
        i3d_labels = torch.ones((i3d.size(0), i3d.size(1))).long() * -1
        video_mask = torch.cat([torch.zeros((i3d.size(0), i3d.size(1))), torch.ones(lm_labels.size())], 1)
        reply_mask = torch.zeros(video_mask.size())
        lm_labels = torch.cat([i3d_labels, lm_labels], dim=1)# video_mask中0代表视频，1代表文字
        # input_ids代表text输入，token_type_ids代表text类型，lm_labels代表除答案外都是-1，i3d代表video的字符
        # video_mask前面代i3d字符，后面代表input_ids字符，分别是01,reply_mask代表视频mask
        return input_ids, token_type_ids, lm_labels, input_mask, i3d, video_mask, reply_mask
    else:
        return input_ids, token_type_ids, lm_labels, input_mask


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(his_nltk, caption, history, reply, tokenizer, with_eos=True, video=False, drop_caption=False, train=True):
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    bos, eos, speaker1, speaker2, cap = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-3])
    # bos:50257 eos:50258 cap:50262 speaker1:50259 speaker2:50260
    if not drop_caption:
        instance = {}
        sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])]
        sequence = [[cap] + sequence[0] + [eos]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

        instance["input_ids"] = list(chain(*sequence))# speaker1为50259,speaker2为50260，顺序是先259后260
        instance["token_type_ids"] = [cap] * len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        if video and train:# 查看sequence中是否有sequence参与
            #instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
    else:
        instance = {}
        sequence = history + [reply + ([eos] if with_eos else [])]
        for i,value in enumerate(his_nltk):
            if(len(history[i]) != len(his_nltk[i])):
                his_nltk[i] = [0] * len(history[i])
            his_nltk[i] = [0] + his_nltk[i]
        sequence_nltk = his_nltk + [[0]*(len(reply)+1) + ([0] if with_eos else [])]
        sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]
        instance["input_ids"] = list(chain(*sequence))
        instance["sequence_nltk"] = list(chain(*sequence_nltk))
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        if video:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]

    return instance, sequence


