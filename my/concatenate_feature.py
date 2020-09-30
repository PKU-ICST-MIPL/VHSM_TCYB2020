# -*- coding: utf-8 -*- 
import tensorflow as tf
def cal_concatenate_representation(passage_rep,question_rep):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    batch_size = tf.shape(passage_rep)[0]
    dim = passage_rep.get_shape().as_list()[2]
    def single_instance(x):
        p = x[0]
        q = x[1]
        # p:[passage_len,dim],q:[question_len,dim]
        p_shape = tf.shape(p)
        q_shape = tf.shape(q)

        tile_p = tf.tile(p, [1, q_shape[0]])  
        tile_p = tf.reshape(tile_p,[-1,p_shape[1]])
        tile_q = tf.tile(q, [p_shape[0], 1])
        p_q = tf.concat(axis = 1 ,values = [tile_p,tile_q])
        return p_q
    elems = (passage_rep,question_rep)
    concatenate_res = tf.map_fn(single_instance,elems,dtype = tf.float32)
    concatenate_res = tf.reshape(concatenate_res,[batch_size,-1,dim*2])
    return concatenate_res,2*dim


def cal_concatenate_representation_new(passage_rep,question_rep):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    batch_size = tf.shape(passage_rep)[0]
    dim = passage_rep.get_shape().as_list()[2]

    passage_shape = tf.shape(passage_rep)
    question_shape = tf.shape(question_rep)

    tile_question = tf.tile(question_rep,[1,passage_shape[1],1])
    tile_passage = tf.tile(passage_rep,[1,1,question_shape[1]])
    tile_passage = tf.reshape(tile_passage,[batch_size,-1,dim])

    concatenate_res = tf.concat(axis = 2, values = [tile_passage,tile_question])

    return concatenate_res, 2*dim

