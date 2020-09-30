#-*- coding: UTF-8 -*-   
import numpy as np
import re
import json
import os
import tensorflow as tf
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)] # zgwang: starting point of each batch

def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in xrange(dim1_size):
        cur_in_vals = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
        out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
    return out_val

def pad_3d_vals(in_vals, dim1_size, dim2_size, dim3_size, dtype=np.int32):
    out_val = np.zeros((dim1_size, dim2_size, dim3_size), dtype=dtype)
    if dim1_size > len(in_vals): dim1_size = len(in_vals)
    for i in xrange(dim1_size):
        in_vals_i = in_vals[i]
        cur_dim2_size = dim2_size
        if cur_dim2_size > len(in_vals_i): cur_dim2_size = len(in_vals_i)
        for j in xrange(cur_dim2_size):
            in_vals_ij = in_vals_i[j]
            cur_dim3_size = dim3_size
            if cur_dim3_size > len(in_vals_ij): cur_dim3_size = len(in_vals_ij)
            out_val[i, j, :cur_dim3_size] = in_vals_ij[:cur_dim3_size]
    return out_val



def construct_one_hot_feature_tensor(sequences, left_padding_and_cropping_pairs, dim, batch_max_sentence_length,column_size=None, dtype=np.int32):
    """
    sequences: [[(idx, val)... ()]...[]]
    left_padding_and_cropping_pairs: [[(0,0)...] ... []]
    """
    size = batch_max_sentence_length
    tensor_list = []
    for sequence, pad_crop_pair in zip(sequences, left_padding_and_cropping_pairs):
        left_padding, left_cropping = pad_crop_pair
        if dim == 1:
            vec = np.zeros((size))
            for num in sequence:
                if num + left_padding - left_cropping < size and num + left_padding - left_cropping >= 0:
                    vec[num + left_padding - left_cropping] = 1
            tensor_list.append(vec)
        elif dim == 2:
            assert column_size
            mtrx = np.zeros((size, column_size))
            for row, col in sequence:
                if row + left_padding - left_cropping < size and row + left_padding - left_cropping >= 0 and col < column_size:
                    mtrx[row + left_padding - left_cropping, col] = 1
            tensor_list.append(mtrx)

        else:
            raise NotImplementedError

    return np.array(tensor_list, dtype=dtype)


def read_all_instances(inpath, word_vocab=None, label_vocab=None, char_vocab=None, max_sent_length=100,
                       max_char_per_word=10, isLower=True):
    instances = []
    infile = open(inpath, 'rt')
    idx = -1
    for line in infile:
        #idx += 1
        line = line.decode('utf-8').strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = items[1].strip()
        sentence2 = items[2].strip()
        imageid = items[3]
        pairID = items[4]
        cur_ID = "{}".format(idx)
        if len(items)>=4: cur_ID = items[3]
        if isLower:
            sentence1 = sentence1.lower()
            sentence2 = sentence2.lower()
        if label_vocab is not None:
            label_id = label_vocab.getIndex(label)
            if label_id >= label_vocab.vocab_size: label_id = 0
        else:
            label_id = int(label)
        word_idx_1 = word_vocab.to_index_sequence(sentence1)
        word_idx_2 = word_vocab.to_index_sequence(sentence2)
        if char_vocab is not None:
            char_matrix_idx_1 = char_vocab.to_character_matrix(sentence1, max_char_per_word=max_char_per_word)
            char_matrix_idx_2 = char_vocab.to_character_matrix(sentence2, max_char_per_word=max_char_per_word)
        else:
            char_matrix_idx_1 = None
            char_matrix_idx_2 = None
        if len(word_idx_1) > max_sent_length:
            word_idx_1 = word_idx_1[:max_sent_length]
            if char_vocab is not None: char_matrix_idx_1 = char_matrix_idx_1[:max_sent_length]
        if len(word_idx_2) > max_sent_length:
            word_idx_2 = word_idx_2[:max_sent_length]
            if char_vocab is not None: char_matrix_idx_2 = char_matrix_idx_2[:max_sent_length]
        image_name = "".join(imageid)
        if "jpg" in image_name:
            idx += 1
            instances.append((label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2, cur_ID,imageid,pairID))
    infile.close()
    return instances

def load_shared_content(fh, shared_content):
    for line in fh:
        row = line.rstrip().split("\t")
        key = row[0]
        value = json.loads(row[1])
        shared_content[key] = value


def load_mnli_shared_content():
    shared_file_exist = False
    
    shared_path = "./data/snli/shared.jsonl"
    
    print(shared_path)
    if os.path.isfile(shared_path):
        shared_file_exist = True
    assert shared_file_exist
    
    with open(shared_path) as f:
        shared_content = {}
        load_shared_content(f, shared_content)
    return shared_content

shared_content = load_mnli_shared_content()

class SentenceMatchDataStream(object):
    def __init__(self, inpath, word_vocab=None, char_vocab=None, label_vocab=None,
                 isShuffle=False, isLoop=False, isSort=True, options=None):
        instances = read_all_instances(inpath, word_vocab=word_vocab, label_vocab=label_vocab,
                    char_vocab=char_vocab, max_sent_length=options.max_sent_length, max_char_per_word=options.max_char_per_word,
                                       isLower=options.isLower)

        # sort instances based on sentence length
        if isSort: instances = sorted(instances, key=lambda instance: (len(instance[4]), len(instance[5]))) # sort instances based on length
        self.num_instances = len(instances)
        
        # distribute into different buckets
        batch_spans = make_batches(self.num_instances, options.batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(instances[i])
            cur_batch = InstanceBatch(cur_instances, with_char=options.with_char)
            self.batches.append(cur_batch)

        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array) 
        self.isLoop = isLoop
        self.cur_pointer = 0
    
    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array) 
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def shuffle(self):
        if self.isShuffle: np.random.shuffle(self.index_array)

    def reset(self):
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[self.index_array[i]]

class InstanceBatch(object):
    def __init__(self, instances, with_char=False):
        self.instances = instances
        self.batch_size = len(instances)
        self.question_len = 0
        self.passage_len = 0
        self.matching_len = 0

        self.question_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_question_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
        self.passage_lengths = []  # tf.placeholder(tf.int32, [None])
        self.in_passage_words = []  # tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
        self.matching_lengths = []
        self.label_truth = []  # [batch_size]
        self.imageid = []
        self.pairID = []
        self.premise_exact_match = []
        self.hypothesis_exact_match = []

        if with_char:
            self.in_question_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, question_len, q_char_len]
            self.question_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, question_len]
            self.in_passage_chars = [] # tf.placeholder(tf.int32, [None, None, None])  # [batch_size, passage_len, p_char_len]
            self.passage_char_lengths = [] # tf.placeholder(tf.int32, [None, None])  # [batch_size, passage_len]
        
        for (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2, cur_ID,imageid,pairID) in instances:
            cur_question_length = len(word_idx_1)
            cur_passage_length = len(word_idx_2)
            if self.question_len < cur_question_length: self.question_len = cur_question_length
            if self.passage_len < cur_passage_length: self.passage_len = cur_passage_length
            self.question_lengths.append(cur_question_length)
            self.in_question_words.append(word_idx_1)
            self.passage_lengths.append(cur_passage_length)
            self.matching_lengths.append(cur_question_length*cur_passage_length)
            self.in_passage_words.append(word_idx_2)
            self.label_truth.append(label_id)
            self.imageid.append(imageid)
            self.pairID.append(pairID)
            if with_char:
                self.in_question_chars.append(char_matrix_idx_1)
                self.in_passage_chars.append(char_matrix_idx_2)
                self.question_char_lengths.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_1])
                self.passage_char_lengths.append([len(cur_char_idx) for cur_char_idx in char_matrix_idx_2])
        
        pairIDs =[]
        for (label, sentence1, sentence2, label_id, word_idx_1, word_idx_2, char_matrix_idx_1, char_matrix_idx_2, cur_ID,imageid,pairID) in instances:
           pairIDs.append(pairID)
        pairIDs = np.array([str(pairIDs[i]) for i in range(len(instances))])
        premise_pad_crop_pair = hypothesis_pad_crop_pair = [(0,0)] * len(instances)

        premise_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence1_token_exact_match_with_s2"][:] for i in range(len(instances))], premise_pad_crop_pair, 1,self.question_len)
        hypothesis_exact_match = construct_one_hot_feature_tensor([shared_content[pairIDs[i]]["sentence2_token_exact_match_with_s1"][:] for i in range(len(instances))], hypothesis_pad_crop_pair, 1,self.passage_len)
        self.premise_exact_match = np.expand_dims(premise_exact_match, 2)
        self.hypothesis_exact_match = np.expand_dims(hypothesis_exact_match, 2)
        
        

        # padding all value into np arrays
        self.question_lengths = np.array(self.question_lengths, dtype=np.int32)
        self.in_question_words = pad_2d_vals(self.in_question_words, self.batch_size, self.question_len, dtype=np.int32)
        self.passage_lengths = np.array(self.passage_lengths, dtype=np.int32)
        self.in_passage_words = pad_2d_vals(self.in_passage_words, self.batch_size, self.passage_len, dtype=np.int32)
        self.matching_lengths = np.array(self.matching_lengths, dtype = np.int32)
        self.label_truth = np.array(self.label_truth, dtype=np.int32)
        if with_char:
            max_char_length1 = np.max([np.max(aa) for aa in self.question_char_lengths])
            self.in_question_chars = pad_3d_vals(self.in_question_chars, self.batch_size,  self.question_len,
                                                   max_char_length1, dtype=np.int32)
            max_char_length2 = np.max([np.max(aa) for aa in self.passage_char_lengths])
            self.in_passage_chars = pad_3d_vals(self.in_passage_chars, self.batch_size,  self.passage_len,
                                                max_char_length2, dtype=np.int32)

            self.question_char_lengths = pad_2d_vals(self.question_char_lengths, self.batch_size,  self.question_len)
            self.passage_char_lengths = pad_2d_vals(self.passage_char_lengths, self.batch_size,  self.passage_len)
