# -*- coding: utf-8 -*-  
import tensorflow as tf
from base_module.nn import softsel, get_logits,fuse_gate
import numpy as np

def share_attention(image, text, image_mask=None, text_mask=None, scope=None):
     with tf.variable_scope("share_attention"):
         batch_size = tf.shape(image)[0]
         len_image = tf.shape(image)[1]
         len_text = tf.shape(text)[1]
         feature_dim = tf.shape(image)[2]
         
         w_share = tf.get_variable("w_share",[200,1],dtype = tf.float32)
         atten_val_image = tf.matmul(tf.reshape(image,[batch_size*len_image, feature_dim]),w_share)  #[batch_size*len_image,1]
         atten_val_image = tf.reshape(atten_val_image,[batch_size,len_image])
         atten_val_text = tf.matmul(tf.reshape(text,[batch_size*len_text, feature_dim]),w_share)  #[batch_size*len_text,1]
         atten_val_text = tf.reshape(atten_val_text,[batch_size,len_text])
         att_val_image_aug =  tf.tile(tf.expand_dims(atten_val_image,2),[1,1,feature_dim])
         att_val_text_aug  =  tf.tile(tf.expand_dims(atten_val_text,2),[1,1,feature_dim])
 
         output_image = tf.multiply(image,att_val_image_aug)
         output_text = tf.multiply(text,att_val_text_aug)
         return(output_image,output_text) 

def bi_attention(p, h, p_mask=None, h_mask=None, scope=None): #[N, L, 2d]
     with tf.variable_scope(scope or "bi_attention"):
         is_train = True
         PL = tf.shape(p)[1]
         HL = tf.shape(h)[1]
         p_aug = tf.tile(tf.expand_dims(p, 2), [1,1,HL,1])
         h_aug = tf.tile(tf.expand_dims(h, 1), [1,PL,1,1]) #[N, PL, HL, 2d]


         if p_mask is None:
             ph_mask = None
         else:
             p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
             h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
             ph_mask = p_mask_aug & h_mask_aug


         h_logits = get_logits([p_aug, h_aug], None, True, wd=0, mask=ph_mask,
                           is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
         h_a = softsel(h_aug, h_logits)
         p_a = softsel(p, tf.reduce_max(h_logits, 2))  # [N, 2d]
         p_a = tf.tile(tf.expand_dims(p_a, 1), [1, PL, 1]) # 

         return h_a, p_a

def self_attention(is_train,p, p_mask=None, scope=None): #[N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        #PL = p.get_shape().as_list()[1]
        #dim = p.get_shape().as_list()[-1]
        PL = tf.shape(p)[1]
        dim = tf.shape(p)[2]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1,1,PL,1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1,PL,1,1]) #[N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2


        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=0.0, mask=self_mask,
                              is_train=is_train, func="mul_linear", scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits)

        return self_att


def self_attention_layer(is_train,p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        #PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(is_train,p, p_mask=p_mask)

        #print("self_att shape")
        #print(self_att.get_shape())

        p0 = fuse_gate(is_train, p, self_att, scope="self_att_fuse_gate")

        return p0

 
