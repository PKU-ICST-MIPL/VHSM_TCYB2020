# -*- coding: utf-8 -*-  
import tensorflow as tf
def gate_attention(in_question_repres,in_passage_repres,image_feature,batch_size):
    shape_q = tf.shape(in_question_repres)
    shape_p = tf.shape(in_passage_repres)
    shape_img = tf.shape(image_feature)
    batch_size = shape_q[0]
    q_length = shape_q[1]
    p_length = shape_p[1]
    img_length = shape_img[1]
    dim_q = shape_q[2]
    dim_p = shape_p[2]
    dim_img = shape_img[2]
    in_val_q = tf.reshape(in_question_repres,[batch_size*q_length,dim_q])
    in_val_p = tf.reshape(in_passage_repres,[batch_size*p_length,dim_p])
    in_val_img = tf.reshape(image_feature,[batch_size*img_length,dim_img])

    #gate_dim = 200
    gate_dim = in_question_repres.get_shape().as_list()[2]
    with tf.variable_scope("gate_attention_q"):
        gate_w = tf.get_variable("gate_w",[gate_dim,gate_dim],dtype = tf.float32)
        gate_b = tf.get_variable("gate_b",[gate_dim],dtype = tf.float32)
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val_q,gate_w,gate_b))
        outputs = in_val_q*gate
        in_questiuon_repres = tf.reshape(outputs,[batch_size,q_length,dim_q])

    with tf.variable_scope("gate_attention_p"):
        gate_w = tf.get_variable("gate_w",[gate_dim,gate_dim],dtype = tf.float32)
        gate_b = tf.get_variable("gate_b",[gate_dim],dtype = tf.float32)
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val_p,gate_w,gate_b))
        outputs = in_val_p*gate
        in_passage_repres = tf.reshape(outputs,[batch_size,p_length,dim_p])

    with tf.variable_scope("gate_attention_img"):
        gate_w = tf.get_variable("gate_w",[gate_dim,gate_dim],dtype = tf.float32)
        gate_b = tf.get_variable("gate_b",[gate_dim],dtype = tf.float32)
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val_img,gate_w,gate_b))
        outputs = in_val_img*gate
        image_feature = tf.reshape(outputs,[batch_size,img_length,dim_img])
       
    return (in_question_repres,in_passage_repres,image_feature)

def concatenate_gate_attention(p_q, scope):
    shape= tf.shape(p_q)
    batch_size = shape[0]
    length = shape[1]
    dim = shape[2]
    #gate_dim = tf.reshape(dim, [])  #to scalar
    gate_dim=p_q.get_shape().as_list()[2]
    in_p_q = tf.reshape(p_q,[batch_size*length,dim])
    #gate_dim = 400 
    with tf.variable_scope(scope):
        gate_w = tf.get_variable("gate_w",[gate_dim,gate_dim],dtype = tf.float32)
        gate_b = tf.get_variable("gate_b",[gate_dim],dtype = tf.float32)
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_p_q,gate_w,gate_b))
        outputs = in_p_q*gate
        out_p_q = tf.reshape(outputs,[batch_size,length,dim])
    return out_p_q
