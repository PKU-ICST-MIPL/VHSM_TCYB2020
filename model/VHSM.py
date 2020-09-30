#-*- coding: UTF-8 -*-   
import tensorflow as tf
import sys
sys.path.append("..")
from utils import layer_utils
from utils import match_utils
from my import gate_layer
from my import concatenate_feature
from my import attention
from my import multihead_attention
import json
import numpy as np
class VHSM(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, options=None, global_step=None):
        self.options = options
        self.create_placeholders()
        self.create_model_graph(num_classes, word_vocab, char_vocab, is_training, global_step=global_step)

    def create_placeholders(self):
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])

        self.matching_lengths = tf.placeholder(tf.int32,[None])

        self.image_lengths = tf.placeholder(tf.int32,[None])
        self.truth = tf.placeholder(tf.int32, [None]) # [batch_size]
        self.in_question_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, question_len]
        self.in_passage_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, passage_len]
        self.image_feature = tf.placeholder(tf.float32, [None,25088])
        self.premise_exact_match = tf.placeholder(tf.float32,[None,None,1])
        self.hypothesis_exact_match = tf.placeholder(tf.float32,[None,None,1])
        self.is_train = tf.placeholder('bool', [], name='is_train')

        if self.options.with_char:
            self.question_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]

    
    def create_feed_dict(self, cur_batch, image, is_training=False):
        sequences = cur_batch.imageid
        tensor_list = []
        
        for sequence in zip(sequences):
            sequence =  "".join(sequence)
            tensor_list.append(image[sequence])
        cur_image_feature =  np.array(tensor_list, dtype=np.float32)
        size = len(sequences)
        image_lengths = 49*np.ones([size])
        feed_dict = {
            self.question_lengths: cur_batch.question_lengths,
            self.passage_lengths: cur_batch.passage_lengths,
            self.matching_lengths: cur_batch.matching_lengths,
            self.in_question_words: cur_batch.in_question_words,
            self.in_passage_words: cur_batch.in_passage_words,
            self.truth : cur_batch.label_truth,
            self.image_feature : cur_image_feature,
            self.image_lengths : image_lengths,
            self.is_train: True,
            self.premise_exact_match:cur_batch.premise_exact_match,
            self.hypothesis_exact_match:cur_batch.hypothesis_exact_match,
        }

        if self.options.with_char:
            feed_dict[self.question_char_lengths] = cur_batch.question_char_lengths
            feed_dict[self.passage_char_lengths] = cur_batch.passage_char_lengths
            feed_dict[self.in_question_chars] = cur_batch.in_question_chars
            feed_dict[self.in_passage_chars] = cur_batch.in_passage_chars

        return feed_dict


    def create_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        p_q_repres = []
        q_p_repres = []
        image_feature = tf.reshape(self.image_feature, [-1])
        image_feature = tf.reshape(image_feature, [-1, 49,512])
 
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable, 
                                                  initializer=tf.constant(word_vocab.word_vecs), dtype=tf.float32)

            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)  #batch_size*question_len
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim
            
            in_question_repres.append(self.premise_exact_match)
            in_question_repres.append(self.premise_exact_match)
            in_passage_repres.append(self.hypothesis_exact_match)
            in_passage_repres.append(self.hypothesis_exact_match)
            
            input_dim +=2
            

        if options.with_char and char_vocab is not None:
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = char_vocab.word_dim
            self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            quesiton_char_mask = tf.sequence_mask(question_char_lengths, q_char_len, dtype=tf.float32)  # [batch_size*question_len, q_char_len]
            in_question_char_repres = tf.multiply(in_question_char_repres, tf.expand_dims(quesiton_char_mask, axis=-1))


            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            passage_char_mask = tf.sequence_mask(passage_char_lengths, p_char_len, dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
            in_passage_char_repres = tf.multiply(in_passage_char_repres, tf.expand_dims(passage_char_mask, axis=-1))

            (question_char_outputs_fw, question_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_question_char_repres, options.char_lstm_dim,
                    input_lengths=question_char_lengths,scope_name="char_lstm", reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            question_char_outputs_fw = layer_utils.collect_final_step_of_lstm(question_char_outputs_fw, question_char_lengths - 1)
            question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
            question_char_outputs = tf.concat(axis=1, values=[question_char_outputs_fw, question_char_outputs_bw])
            question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, 2*options.char_lstm_dim])

            (passage_char_outputs_fw, passage_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_passage_char_repres, options.char_lstm_dim,
                    input_lengths=passage_char_lengths, scope_name="char_lstm", reuse=True,
                    is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            passage_char_outputs_fw = layer_utils.collect_final_step_of_lstm(passage_char_outputs_fw, passage_char_lengths - 1)
            passage_char_outputs_bw = passage_char_outputs_bw[:, 0, :]
            passage_char_outputs = tf.concat(axis=1, values=[passage_char_outputs_fw, passage_char_outputs_bw])
            passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, 2*options.char_lstm_dim])
                
            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)

            input_dim += 2*options.char_lstm_dim
            
            

        in_question_repres = tf.concat(axis=2, values=in_question_repres) # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2, values=in_passage_repres) # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres, (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres, (1 - options.dropout_rate))

        mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]
        image_mask = tf.sequence_mask(self.image_lengths,49,dtype = tf.float32)
        #image_mask = tf.ones([batch_size, 49], dtype = tf.float32)
        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)
        with tf.variable_scope("t_h"):
            (in_question_repres, in_passage_repres) = match_utils.context_layer(in_question_repres, in_passage_repres,
                            self.question_lengths, self.passage_lengths, question_mask, mask, input_dim, is_training, options=options)
        with tf.variable_scope("i_h"):
            image_feature = match_utils.context_layer_image(image_feature,self.image_lengths, image_mask,512, is_training, options=options)


        #a = tf.Variable([1])
        #is_train = tf.cast(a,dtype=tf.bool)
        is_train = self.is_train
        #intra-attention modeling
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            premise_mask = tf.cast(tf.expand_dims(question_mask, -1), tf.float32)
            hypothesis_mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)

            in_question_rerpes = attention.self_attention_layer(is_train,in_question_repres, p_mask=premise_mask, scope="p_self_attention") # [N, len, dim]    
            in_passage_repres = attention.self_attention_layer(is_train,in_passage_repres, p_mask=hypothesis_mask, scope="h_self_attention")

            in_image_repres = attention.self_attention_layer(is_train,image_feature, p_mask=tf.cast(tf.expand_dims(image_mask,-1),tf.float32), scope="i_self_attention")
 

        (p_q_repres, concatenate_dim) = concatenate_feature.cal_concatenate_representation_new(in_question_repres, in_passage_repres)
        (q_p_repres, concatenate_dim) = concatenate_feature.cal_concatenate_representation_new(in_passage_repres, in_question_repres)

        (p_i_repres, concatenate_dim) = concatenate_feature.cal_concatenate_representation_new(in_image_repres,in_passage_repres)
        (i_p_repres, concatenate_dim) = concatenate_feature.cal_concatenate_representation_new(in_passage_repres,in_image_repres)


        out_p_q = multihead_attention.multihead_attention(queries = p_q_repres,
                                       keys=p_q_repres,
                                       num_units=400,
                                       num_heads=1,
                                       dropout_rate=0.2,
                                       is_training=True,
                                       causality=False,
                                       scope="self_attention_p_q")


        out_q_p = multihead_attention.multihead_attention(queries=q_p_repres,
                                       keys=q_p_repres,
                                       num_units=400,
                                       num_heads=1,
                                       dropout_rate=0.2,
                                       is_training=True,
                                       causality=False,
                                       scope="self_attention_q_p")

        out_p_i = gate_layer.concatenate_gate_attention(p_i_repres, scope="gate_p_i")
        out_i_p = gate_layer.concatenate_gate_attention(i_p_repres, scope="gate_i_p") 


        final1 = tf.reduce_mean(out_p_q,1)
        final1_1 = tf.reduce_max(out_p_q,1)
        final2 = tf.reduce_mean(out_q_p,1)
        final2_2 = tf.reduce_max(out_q_p,1)

        final3 = tf.reduce_mean(out_p_i,1)
        final3_1 = tf.reduce_max(out_p_i,1)
        final4 = tf.reduce_mean(out_i_p,1)
        final4_2 = tf.reduce_max(out_i_p,1)


        match_representation = tf.concat(axis = 1, values = [final1,final1_1,final2,final2_2])
        match_representation_2 = tf.concat(axis = 1, values = [final3,final3_1,final4,final4_2])

        match_dim = 4*concatenate_dim    #2*concatenate_dim
        
        w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)
        
        w_2 = tf.get_variable("w_2", [match_dim, match_dim/2], dtype=tf.float32)
        b_2 = tf.get_variable("b_2", [match_dim/2], dtype=tf.float32)
        w_3 = tf.get_variable("w_3", [match_dim/2, num_classes],dtype=tf.float32)
        b_3 = tf.get_variable("b_3", [num_classes],dtype=tf.float32)



        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits_1 = tf.matmul(match_representation, w_0) + b_0
        logits_1 = tf.tanh(logits_1)
        if is_training: logits_1 = tf.nn.dropout(logits_1, (1 - options.dropout_rate))
        logits_1 = tf.matmul(logits_1, w_1) + b_1

        #self.prob = tf.nn.softmax(logits)
        
        logits_2 = tf.matmul(match_representation_2, w_2) + b_2
        logits_2 = tf.tanh(logits_2)
        if is_training: logits_2 = tf.nn.dropout(logits_2, (1 - options.dropout_rate))
        logits_2 = tf.matmul(logits_2, w_3) + b_3

        logits = 0.4*logits_1 + 0.6*logits_2
        self.prob = tf.nn.softmax(logits)


        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))



        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.argmax(self.prob, 1)

        if not is_training: return

        tvars = tf.trainable_variables()
        if self.options.lambda_l2>0.0:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + self.options.lambda_l2 * l2_loss
        
        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)

        grads = layer_utils.compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)

