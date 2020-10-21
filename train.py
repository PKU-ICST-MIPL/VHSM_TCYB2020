# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import json

from utils.vocab_utils import Vocab
from utils.SentenceMatchDataStream import SentenceMatchDataStream
from model.VHSM import VHSM
import utils.namespace_utils as namespace_utils
import my.transfer as transfer


def collect_vocabs(train_path, with_POS=False, with_NER=False):
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()
    infile = open(train_path, 'rt')
    for line in infile:
        line = line.decode('utf-8').strip()
        if line.startswith('-'): continue
        items = re.split("\t", line)
        label = items[0]
        sentence1 = re.split("\\s+",items[1].lower())
        sentence2 = re.split("\\s+",items[2].lower())
        all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS: 
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER: 
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def output_probs(probs, label_vocab):
    out_string = ""
    for i in xrange(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def evaluation(sess, valid_graph, devDataStream, image, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    total = 0
    correct = 0
    for batch_index in xrange(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, image,is_training=True)
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        correct += cur_correct
        if outpath is not None:
            for i in xrange(cur_batch.batch_size):
                (label, sentence1, sentence2, _, _, _, _, _, cur_ID,imageid,pairID) = cur_batch.instances[i]
                result_json[cur_ID] = {
                    "ID": cur_ID,
                    "truth": label,
                    "sent1": sentence1,
                    "sent2": sentence2,
                    "prediction": label_vocab.getWord(predictions[i]),
                    "probs": output_probs(probs[i], label_vocab),
                }
    accuracy = correct / float(total) * 100
    if outpath is not None:
        with open(outpath, 'w') as outfile:
            json.dump(result_json, outfile)
    return accuracy

def train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, options, best_path, image):
    best_accuracy = evaluation(sess, valid_graph, devDataStream,image)
    print('best accuracy restored from pretrained model %.2f' %best_accuracy)
    log = open("./logs/log.txt", "a+")
    for epoch in range(options.max_epochs):
        print('Train in epoch %d' % epoch)
        # training
        trainDataStream.shuffle()
        num_batch = trainDataStream.get_num_batch()
        start_time = time.time()
        total_loss = 0
        total_correct = 0
        total = 0
        for batch_index in xrange(num_batch):  # for each batch
            cur_batch = trainDataStream.get_batch(batch_index)
            feed_dict = train_graph.create_feed_dict(cur_batch, image,is_training=True)
            _, loss_value,correct_value= sess.run([train_graph.train_op, train_graph.loss, train_graph.eval_correct], feed_dict=feed_dict)
            total += cur_batch.batch_size
            total_loss += loss_value
            total_correct += correct_value
            if batch_index % 100 == 0:
                print('{} '.format(batch_index), end="")
                sys.stdout.flush()

        print()
        duration = time.time() - start_time
        print('Epoch %d: loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
        log.write('Epoch %d: loss = %.4f (%.3f sec)   ' % (epoch, total_loss / num_batch, duration))

        # evaluation
        start_time = time.time()
        acc = evaluation(sess, valid_graph, devDataStream,image)
        duration = time.time() - start_time
        print("Dev Accuracy: %.2f" % acc)
        log.write("Accuracy: %.2f   " % acc)
        print('Evaluation time: %.3f sec' % (duration))
        log.write('Evaluation time: %.3f sec\n' % (duration))
        if acc>= best_accuracy:
            best_accuracy = acc
            saver.save(sess, best_path)
    log.close()


def load_image_feature(path):
        f = open(path,'r')
        feature = json.load(f)
        return feature



def main(FLAGS):
    train_path = FLAGS.train_path
    dev_path = FLAGS.dev_path
    word_vec_path = FLAGS.word_vec_path
    log_dir = FLAGS.model_dir
    image_dir = FLAGS.image_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    print("load image feature!")
    image = load_image_feature(image_dir)
    
    path_prefix = log_dir + "/{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    # build vocabs
    word_vocab = Vocab(word_vec_path, fileformat='txt3')

    best_path = path_prefix + '.best.model'
    char_path = path_prefix + ".char_vocab"
    label_path = path_prefix + ".label_vocab"
    has_pre_trained_model = False
    char_vocab = None
    if os.path.exists(best_path + ".index"):
        has_pre_trained_model = True
        print('Loading vocabs from a pre-trained model ...')
        label_vocab = Vocab(label_path, fileformat='txt2')
        if FLAGS.with_char: char_vocab = Vocab(char_path, fileformat='txt2')
    else:
        print('Collecting words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
        print('Number of words: {}'.format(len(all_words)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels,dim=2)
        label_vocab.dump_to_txt2(label_path)

        if FLAGS.with_char:
            print('Number of chars: {}'.format(len(all_chars)))
            char_vocab = Vocab(fileformat='voc', voc=all_chars,dim=FLAGS.char_emb_dim)
            char_vocab.dump_to_txt2(char_path)
        
    print('word_vocab shape is {}'.format(word_vocab.word_vecs.shape))
    num_classes = label_vocab.size()
    print("Number of labels: {}".format(num_classes))
    sys.stdout.flush()

    print('Build SentenceMatchDataStream ... ')
    trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=True, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    sys.stdout.flush()
    devDataStream = SentenceMatchDataStream(dev_path, word_vocab=word_vocab, char_vocab=char_vocab, label_vocab=label_vocab,
                                              isShuffle=False, isLoop=True, isSort=True, options=FLAGS)
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))

    sys.stdout.flush()

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_graph = VHSM(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=True, options=FLAGS, global_step=global_step)

        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_graph = VHSM(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                is_training=False, options=FLAGS)

                
        initializer = tf.global_variables_initializer()
        vars_ = {}
        for var in tf.global_variables():
            if "word_embedding" in var.name: continue
            vars_[var.name.split(":")[0]] = var
        
        if FLAGS.model_transfer:
            transfer.model_transfer(vars_)
        else:
            pass
        saver = tf.train.Saver(vars_)
        
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = True 
        sess = tf.Session(config = config)
        sess.run(initializer)
        if has_pre_trained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

        # training
        train(sess, saver, train_graph, valid_graph, trainDataStream, devDataStream, FLAGS, best_path, image)

def enrich_options(options):
    if not options.__dict__.has_key("in_format"):
        options.__dict__["in_format"] = 'tsv'

    return options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,default="./data/snli/train_new.tsv", help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default="./data/snli/dev_new.tsv",help='Path to the dev set.')
    parser.add_argument('--word_vec_path', type=str, default="./data/snli/wordvec.txt",help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, default="./logs",help='Directory to save model files.')
    parser.add_argument('--suffix', type=str, default='vhsm', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec',default=True)
    parser.add_argument('--isLower',default=True)
    parser.add_argument('--max_sent_length', type=int, default=30, help='Maximum number of words within each sentence.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')

    parser.add_argument('--with_char', default=True, help='With character-composed embeddings.', action='store_true')
    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--char_lstm_dim', type=int, default=105, help='Number of dimension for character-composed embeddings.')
    
    parser.add_argument('--batch_size', type=int, default=100, help='Number of instances in each batch.')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum epochs for training.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout ratio.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--optimize_type',default='adam')   
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--grad_clipper',default=10.0)

    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')

    parser.add_argument('--with_full_match', default=True, help='With full matching.', action='store_true')
    parser.add_argument('--with_maxpool_match', default=False, help='With maxpooling matching', action='store_true')
    parser.add_argument('--with_max_attentive_match', default=False, help='With max attentive matching.', action='store_true')   
    parser.add_argument('--with_attentive_match', default=True, help='With attentive matching', action='store_true')

    parser.add_argument('--with_cosine',default=True)
    parser.add_argument('--with_mp_cosine',default=True)
    parser.add_argument('--cosine_MP_dim',default=5)

    parser.add_argument('--att_dim',default=50)
    parser.add_argument('--att_type',default="symmetric")
  
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--with_highway', default=True, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_match_highway', default=True, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=True, help='Utilize highway layers for aggregation layer.', action='store_true')

    parser.add_argument('--use_cudnn',default=True)

    parser.add_argument('--with_moving_average',default=False)
    parser.add_argument('--model_transfer',default=False)
    parser.add_argument('--image_dir',default="./data/image_feature.jsonl")    
    args, unparsed = parser.parse_known_args()
   
    print(args) 
    FLAGS = args
    sys.stdout.flush()
    
    main(FLAGS)

