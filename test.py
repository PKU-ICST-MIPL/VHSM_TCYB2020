# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import sys
from utils.vocab_utils import Vocab
import utils.namespace_utils as namespace_utils
import json
import tensorflow as tf
from utils.SentenceMatchDataStream import SentenceMatchDataStream
from model.VHSM import VHSM
import random
random.seed(0)
def load_image_feature(path):
        f = open(path,'r')
        feature = json.load(f)
        return feature

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
        feed_dict = valid_graph.create_feed_dict(cur_batch, image,is_training=False)
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
    print(correct, total)
    accuracy = correct / float(total) * 100
    if outpath is not None:
        with open(outpath, 'w') as outfile:
            json.dump(result_json, outfile)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, help='word embedding file for the input file.')

    args, unparsed = parser.parse_known_args()
    
    # load the configuration file
    print('Loading configurations.')
    options = namespace_utils.load_namespace(args.model_prefix + ".config.json")

    if args.word_vec_path is None: args.word_vec_path = options.word_vec_path

    #load image
    print("load image feature!")
    image = load_image_feature("./data/image_feature.jsonl")


    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(args.word_vec_path, fileformat='txt3')
    label_vocab = Vocab(args.model_prefix + ".label_vocab", fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    print('label_vocab: {}'.format(label_vocab.word_vecs.shape))
    num_classes = label_vocab.size()

    if options.with_char:
        char_vocab = Vocab(args.model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    
    print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchDataStream(args.in_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                            label_vocab=label_vocab,
                                            isShuffle=False, isLoop=False, isSort=True, options=options)
    print('Number of instances in devDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in devDataStream: {}'.format(testDataStream.get_num_batch()))
    #sys.stdout.flush()

    best_path = args.model_prefix + ".best.model"

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = VHSM(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  is_training=False, options=options)

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.global_variables():
            print(var.name)
            if "word_embedding" in var.name: continue
            #if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver  = tf.train.Saver(vars_)
 
 
        sess = tf.Session()
        sess.run(initializer)
        print("Restoring model from " + best_path)
        saver.restore(sess, best_path)
        print("DONE!")
        acc = evaluation(sess, valid_graph, testDataStream,image, outpath=args.out_path,
                                              label_vocab=label_vocab)
  
        print(acc)



