#! -*- coding: utf-8 -*-
from keras.layers import *
from keras import backend as K
import tensorflow as tf

class My_LSTM(Layer):

    def __init__(self, units, **kwargs):
        self.units = units 
        super(My_LSTM, self).__init__(**kwargs)

    def build(self, input_shape): 

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.units *4 ),
                                      initializer='glorot_normal',
                                      trainable=True)

        self.recurrent_kernel = self.add_weight(
                                shape=(self.units, self.units * 4),
                                name='recurrent_kernel',
                                initializer='glorot_normal',
                                trainable = True
                                                )
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]


    def step_do(self, step_in, states): 

        x_i = K.dot(step_in, self.kernel_i)
        x_f = K.dot(step_in, self.kernel_f)
        x_c = K.dot(step_in, self.kernel_c)
        x_o = K.dot(step_in, self.kernel_o)
        h_tm1= states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        i = K.hard_sigmoid(x_i + K.dot(h_tm1,self.recurrent_kernel_i))
        f = K.hard_sigmoid(x_f + K.dot(h_tm1,self.recurrent_kernel_f))
        o = K.hard_sigmoid(x_o + K.dot(h_tm1, self.recurrent_kernel_o))
        m =x_c + K.dot(h_tm1,self.recurrent_kernel_c)
        c = f * c_tm1 + i * m

        h =  o * K.tanh(c)
        ch = K.concatenate([c,h])

        return ch, [h,c]

    def call(self, inputs):
        init_states = [tf.zeros((K.shape(inputs)[0],self.units)),tf.zeros((K.shape(inputs)[0],self.units))]
        outputs = K.rnn(self.step_do, inputs, init_states)
        return outputs[1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.units*2)


class MGM(Layer):

    def __init__(self,dmem,**kwargs):
        self.output_dim = dmem
        super(MGM,self).__init__(**kwargs)

    def build(self, input_shape):  
        self.W_Du = self.add_weight(name='W1',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)

        self.W_Dr1 = self.add_weight(name='W2',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)

        self.W_Dr2 = self.add_weight(name='W3',
                                       shape=(input_shape[-1], self.output_dim),
                                       initializer='glorot_normal',
                                       trainable=True)


        self.b_Du = self.add_weight(name='b1',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)
        self.b_Dr1 = self.add_weight(name='b2',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)
        self.b_Dr2 = self.add_weight(name='b3',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)

    def step_do(self,step_in,states):
        r1 = K.softmax(K.dot(step_in,self.W_Dr1) + self.b_Dr1)
        r2 = K.softmax(K.dot(step_in,self.W_Dr2) + self.b_Dr2)
        u_tide = K.dot(step_in,self.W_Du)+self.b_Du
        step_out = r1*states[0]+r2*K.tanh(u_tide)

        return step_out,[step_out]


    def call(self,inputs):
        init_states = [tf.zeros((K.shape(inputs)[0],self.output_dim))]
        outputs = K.rnn(self.step_do, inputs, init_states)
        return outputs[1]


    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.output_dim)


def mfn(reps,concatenate_dim):

    dim = concatenate_dim/2
    ch= My_LSTM(emb)(reps)
    
    c_s =Lambda(lambda x: x[:,:,:dim])(ch)
    h = Lambda(lambda x: x[:,0:-1,dim:])(ch)

    t_1 = Lambda(lambda x: x[:,0:-1,:])(c_s)
    t_2 = Lambda(lambda x: x[:,1:,: ])(c_s)
    c = concatenate([t_1, t_2])

    a = TimeDistributed(Dense(dim*2,activation='softmax'))(c) 
    c_tide = multiply([c, a]) 
    u =  MGM(dim)(c_tide)

    res = concatenate([h,u])

    return res

def DMAN(reps,concatenate_dim):
    dim = concatenate_dim/2
    ch= My_LSTM(dim)(reps)

    c_s =Lambda(lambda x: x[:,:,:dim])(ch)

    t_1 = Lambda(lambda x: x[:,0:-1,:])(c_s)
    t_2 = Lambda(lambda x: x[:,1:,: ])(c_s)
    c = concatenate([t_1, t_2])

    a = TimeDistributed(Dense(dim*2,activation='softmax'))(c)
    c_hat = multiply([c, a])

    u = TimeDistributed(Dense(dim*2))(c_hat)

    return u 
