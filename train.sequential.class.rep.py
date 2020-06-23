# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from load_data import load_CLINC150_with_specific_domain_sequence, load_CLINC150_without_specific_domain

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification

# from transformers.modeling_bert import BertModel
# from transformers.tokenization_bert import BertTokenizer
# from bert_common_functions import store_transformers_models

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'


class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        # self.roberta_single= BertModel.from_pretrained(pretrain_model_dir)
        # self.single_hidden2tag = nn.Linear(bert_hidden_dim, tagset_size)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

        # self.roberta_pair = RobertaModel.from_pretrained(pretrain_model_dir)
        # self.pair_hidden2score = nn.Linear(bert_hidden_dim, 1)

    # def forward(self, batch_single, batch_pairs):
    def forward(self, input_ids, input_mask, input_seg, labels):
        # single_train_input_ids, single_train_input_mask, single_train_segment_ids, single_train_label_ids = batch_single
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1] #(batch, hidden)
        # print('hidden_states_single:', hidden_states_single)
        score_single, last_reps, bias = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)

        # pair_train_input_ids, pair_train_input_mask, pair_train_segment_ids, pair_train_label_ids = batch_pairs
        # # outputs_pair = self.roberta_pair(pair_train_input_ids, pair_train_input_mask,pair_train_segment_ids)
        # '''roberta does not use token_type, i.e., setment_ids'''
        # outputs_pair = self.roberta_pair(pair_train_input_ids, pair_train_input_mask,None)
        # # print('outputs_pair shape:', outputs_pair[2].shape)
        # hidden_states_pair = outputs_pair[1] #(batch*tagset, hidden)
        # score_pair = self.pair_hidden2score(hidden_states_pair).view(-1, self.tagset_size) #(batch, tagset)

        logits = score_single#+score_pair
        return logits, last_reps, bias

class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x_rep = self.dropout(x)
        x = self.out_proj(x_rep)
        return x, x_rep, self.out_proj.bias

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""


    def get_RTE_as_train(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "train-"+str(line_co-1)
                text_a = line[1].strip()
                text_b = line[2].strip()
                label = 'entailment' if line[3].strip()=='entailment' else 'not_entailment' #["entailment", "not_entailment"]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
            # if line_co > 20000:
            #     break
        readfile.close()
        print('loaded  size:', line_co)
        return examples

    def get_RTE_as_dev(self, filename):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "dev-"+str(line_co-1)
                text_a = line[1].strip()
                text_b = line[2].strip()
                # label = line[3].strip() #["entailment", "not_entailment"]
                label = 'entailment' if line[3].strip()=='entailment' else 'not_entailment'
                # label = 'entailment'  if line[3] == 'entailment' else 'neutral'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            line_co+=1
            # if line_co > 20000:
            #     break
        readfile.close()
        print('loaded  size:', line_co-1)
        return examples

    def get_RTE_as_test(self, filename):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==3:
                guid = "test-"+str(line_co)
                text_a = line[1]
                text_b = line[2]
                '''for RTE, we currently only choose randomly two labels in the set, in prediction we then decide the predicted labels'''
                label = 'entailment'  if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1

        readfile.close()
        print('loaded test size:', line_co)
        return examples

    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()










def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_data_aug",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--kshot',
                        type=int,
                        default=5,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    # label_list = processor.get_labels() #["entailment", "neutral", "contradiction"]
    # label_list = ['How_do_I_create_a_profile_v4', 'Profile_Switch_v4', 'Deactivate_Active_Devices_v4', 'Ads_on_Hulu_v4', 'Watching_Hulu_with_Live_TV_v4', 'Hulu_Costs_and_Commitments_v4', 'offline_downloads_v4', 'womens_world_cup_v5', 'forgot_username_v4', 'confirm_account_cancellation_v4', 'Devices_to_Watch_HBO_on_v4', 'remove_add_on_v4', 'Internet_Speed_for_HD_and_4K_v4', 'roku_related_questions_v4', 'amazon_related_questions_v4', 'Clear_Browser_Cache_v4', 'ads_on_ad_free_plan_v4', 'inappropriate_ads_v4', 'itunes_related_questions_v4', 'Internet_Speed_Recommendations_v4', 'NBA_Basketball_v5', 'unexpected_charges_v4', 'change_billing_date_v4', 'NFL_on_Hulu_v5', 'How_to_delete_a_profile_v4', 'Devices_to_Watch_Hulu_on_v4', 'Manage_your_Hulu_subscription_v4', 'cancel_hulu_account_v4', 'disney_bundle_v4', 'payment_issues_v4', 'home_network_location_v4', 'Main_Menu_v4', 'Resetting_Hulu_Password_v4', 'Update_Payment_v4', 'I_need_general_troubleshooting_help_v4', 'What_is_Hulu_v4', 'sprint_related_questions_v4', 'Log_into_TV_with_activation_code_v4', 'Game_of_Thrones_v4', 'video_playback_issues_v4', 'How_to_edit_a_profile_v4', 'Watchlist_Remove_Video_v4', 'spotify_related_questions_v4', 'Deactivate_Login_Sessions_v4', 'Transfer_to_Agent_v4', 'Use_Hulu_Internationally_v4']

    meta_train_examples, meta_dev_examples, meta_test_examples, meta_label_list = load_CLINC150_without_specific_domain('banking')
    train_examples, dev_examples, eval_examples, finetune_label_list = load_CLINC150_with_specific_domain_sequence('banking', args.kshot, augment=args.do_data_aug)
    label_list=finetune_label_list+meta_label_list+['oos']
    assert len(label_list) ==  15*10+1
    num_labels = len(label_list)-1
    assert num_labels == 15*10

    # train_examples = None
    # num_train_optimization_steps = None
    # if args.do_train:
    #     num_train_optimization_steps = int(
    #         len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    #     if args.local_rank != -1:
    #         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank))

    # pretrain_model_dir = 'roberta-large-mnli' #'roberta-large' , 'roberta-large-mnli'
    # pretrain_model_dir = '/export/home/Dataset/BERT_pretrained_mine/crossdataentail/trainMNLItestRTE/0.8772563176895307'
    model = RobertaForSequenceClassification(num_labels)


    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    # tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0
    if args.do_train:
        meta_train_features = convert_examples_to_features(
            meta_train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)


        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        '''load dev set'''
        # dev_examples = processor.get_RTE_as_dev('/export/home/Dataset/glue_data/RTE/dev.tsv')
        # dev_examples = get_data_hulu('dev')
        dev_features = convert_examples_to_features(
            dev_examples, finetune_label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        dev_all_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
        dev_all_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
        dev_all_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
        dev_all_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

        dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)


        '''load test set'''
        # eval_examples = processor.get_RTE_as_test('/export/home/Dataset/RTE/test_RTE_1235.txt')
        # eval_examples = get_data_hulu('test')
        eval_features = convert_examples_to_features(
            eval_examples, finetune_label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

        eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        # logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in meta_train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in meta_train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in meta_train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in meta_train_features], dtype=torch.long)

        meta_train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        meta_train_sampler = RandomSampler(meta_train_data)
        meta_train_dataloader = DataLoader(meta_train_data, sampler=meta_train_sampler, batch_size=args.train_batch_size*10)


        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        '''support labeled examples in order, group in kshot size'''
        support_sampler = SequentialSampler(train_data)
        support_dataloader = DataLoader(train_data, sampler=support_sampler, batch_size=args.kshot)


        iter_co = 0
        class_reps_history = []
        class_bias_history = []
        '''first train on meta_train tasks'''
        for meta_epoch_i in trange(1, desc="metaEpoch"):
            for step, batch in enumerate(tqdm(meta_train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits,_,_ = model(input_ids, input_mask, None, labels=None)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print('meta_epoch_i', meta_epoch_i, ' loss:', loss)

        '''get class representation after pretraining'''
        model.eval()
        last_reps_list = []
        for input_ids, input_mask, segment_ids, label_ids in support_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            # gold_label_ids+=list(label_ids.detach().cpu().numpy())

            with torch.no_grad():
                logits, last_reps, bias = model(input_ids, input_mask, None, labels=None)
            last_reps_list.append(last_reps.mean(dim=0, keepdim=True)) #(1, 1024)
        class_reps_pretraining = torch.cat(last_reps_list, dim=0) #(15, 1024)
        '''second finetune'''
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits,_,_ = model(input_ids, input_mask, None, labels=None)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                iter_co+=1
                # if iter_co %20==0:
                if iter_co % len(train_dataloader)==0:
                    model.eval()
                    '''first get the class representation'''
                    print('\t\t infering the class representation at current step...')
                    last_reps_list = []
                    for input_ids, input_mask, segment_ids, label_ids in support_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        # gold_label_ids+=list(label_ids.detach().cpu().numpy())

                        with torch.no_grad():
                            logits, last_reps, bias = model(input_ids, input_mask, None, labels=None)
                        # print('bias:', bias)
                        # exit(0)
                        last_reps_list.append(last_reps.mean(dim=0, keepdim=True)) #(1, 1024)
                    class_reps_finetune = torch.cat(last_reps_list, dim=0) #(15, 1024)
                    bias_finetune = bias[:15] #the first 15 classes is for the target domain

                    # class_reps_history.append(class_reps_i)
                    # class_bias_history.append(bias)
                    # if len(class_reps_history)>5:
                    #     class_reps_history = class_reps_history[-5:]
                    #     class_bias_history = class_bias_history[-5:]
                    #
                    # class_representation_matrix = torch.cat(class_reps_history[-5:], dim=0) #(15*5, 1024)
                    # class_bias_vector = torch.cat(class_bias_history[-5:]) #15*5
                    '''
                    start evaluate on dev set after this epoch
                    '''
                    for idd, dev_or_test_dataloader in enumerate([dev_dataloader, eval_dataloader]):
                        if idd == 0:
                            logger.info("***** Running dev *****")
                            logger.info("  Num examples = %d", len(dev_examples))
                        else:
                            logger.info("***** Running test *****")
                            logger.info("  Num examples = %d", len(eval_examples))
                        # logger.info("  Batch size = %d", args.eval_batch_size)

                        eval_loss = 0
                        nb_eval_steps = 0
                        preds = []
                        gold_label_ids = []
                        # print('Evaluating...')
                        for input_ids, input_mask, segment_ids, label_ids in dev_or_test_dataloader:
                            input_ids = input_ids.to(device)
                            input_mask = input_mask.to(device)
                            segment_ids = segment_ids.to(device)
                            label_ids = label_ids.to(device)
                            gold_label_ids+=list(label_ids.detach().cpu().numpy())

                            with torch.no_grad():
                                logits_LR, reps_batch, _ = model(input_ids, input_mask, None, labels=None)
                            # logits = logits[0]

                            '''pretraining logits'''
                            raw_similarity_scores = torch.mm(reps_batch,torch.transpose(class_reps_pretraining, 0,1)) #(batch, 15)
                            # print('raw_similarity_scores shaoe:', raw_similarity_scores.shape)
                            # print('bias_finetune:', bias_finetune.shape)
                            biased_similarity_scores = raw_similarity_scores+bias_finetune.view(-1, raw_similarity_scores.shape[1])
                            logits_pretrain = torch.max(biased_similarity_scores.view(args.eval_batch_size, -1, len(finetune_label_list)), dim=1)[0] #(batch, #class)
                            '''finetune logits'''
                            raw_similarity_scores = torch.mm(reps_batch,torch.transpose(class_reps_finetune, 0,1)) #(batch, 15*history)
                            biased_similarity_scores = raw_similarity_scores+bias_finetune.view(-1, raw_similarity_scores.shape[1])
                            logits_finetune = torch.max(biased_similarity_scores.view(args.eval_batch_size, -1, len(finetune_label_list)), dim=1)[0] #(batch, #class)

                            logits = logits_pretrain+logits_finetune
                            # logits = (1-0.9)*logits+0.9*logits_LR


                            # loss_fct = CrossEntropyLoss()
                            # tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                            # eval_loss += tmp_eval_loss.mean().item()
                            # nb_eval_steps += 1
                            if len(preds) == 0:
                                preds.append(logits.detach().cpu().numpy())
                            else:
                                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                        # eval_loss = eval_loss / nb_eval_steps
                        preds = preds[0]

                        '''
                        preds: size*3 ["entailment", "neutral", "contradiction"]
                        wenpeng added a softxmax so that each row is a prob vec
                        '''
                        pred_probs = softmax(preds,axis=1)
                        pred_label_ids = list(np.argmax(pred_probs, axis=1))
                        max_probs = list(np.max(pred_probs, axis=1))
                        for i, prob_i in enumerate(max_probs):
                            if prob_i < (1/15)*2:
                                pred_label_ids[i] = len(label_list)-1 #oos indice

                        pred_oos = [1 for x in pred_label_ids if x == len(label_list)-1 else 0]
                        gold_oos = [1 for x in gold_label_ids if x == len(label_list)-1 else 0]

                        overlap_oos = 0
                        for i in range(len(pred_oos)):
                            if gold_oos[i] == 1 and  pred_oos[i] ==1:
                                overlap_oos +=1
                        recall_oos = overlap_oos/sum(gold_oos)
                        precision_oos = overlap_oos/sum(pred_oos)
                        f1_oos = 2*recall_oos*precision_oos/(recall_oos+precision_oos)





                        # print('pred_label_ids:', pred_label_ids)

                        gold_label_ids = gold_label_ids
                        assert len(pred_label_ids) == len(gold_label_ids)
                        hit_co = 0
                        sum_co = 0
                        for k in range(len(pred_label_ids)):
                            if gold_label_ids[k]!=len(label_list)-1:
                                sum_co+=1
                                if pred_label_ids[k] == gold_label_ids[k]:
                                    hit_co +=1
                        if idd == 0:
                            assert sum_co == 15*20
                        else:
                            assert sum_co == 15*30
                        test_acc = hit_co/sum_co

                        if idd == 0: # this is dev
                            if test_acc > max_dev_acc:
                                max_dev_acc = test_acc
                                print('\ndev acc:', test_acc, ' max_dev_acc:', max_dev_acc, 'OOS:',recall_oos, precision_oos, f1_oos, '\n')

                            else:
                                print('\ndev acc:', test_acc, ' max_dev_acc:', max_dev_acc, 'OOS:',recall_oos, precision_oos, f1_oos, '\n')
                                break
                        else: # this is test
                            if test_acc > max_test_acc:
                                max_test_acc = test_acc
                            print('\ttest acc:', test_acc, ' max_test_acc:', max_test_acc, 'OOS:',recall_oos, precision_oos, f1_oos, '\n')
                            # print('\ntest acc:', test_acc, ' max_test_acc:', max_test_acc, '\n')




if __name__ == "__main__":
    main()
    '''
    because classifier not initlized, so smaller learning rate 2e-6
    and fine-tune roberta-large needs more epochs
    '''
# CUDA_VISIBLE_DEVICES=6 python -u train.sequential.class.rep.py --task_name rte --do_train --do_lower_case --num_train_epochs 200 --data_dir '' --output_dir '' --train_batch_size 5 --eval_batch_size 5 --learning_rate 5e-6 --max_seq_length 20 --seed 42 --kshot 1 --do_data_aug
