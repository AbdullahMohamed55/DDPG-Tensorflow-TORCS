import tensorflow as tf
import numpy as np
import time

state_dim= 29
action_dim = 3

class Critic(object):


    def __init__(self,sess,BATCH_SIZE,TAU,LEARNING_RATE):
    # def __init__(self,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        # tf.reset_default_graph()
        # self.sess =  tf.InteractiveSession()
        
        self.init_input()
        self.net,self.weights = self.init_network("CriticNetwork")
        self.target_net,self.target_weights = self.init_network("CriticTarget_Network")
        self.init_ops()
        self.assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            self.assigns.append(tf.assign(w_target, self.TAU * w_agent+ (1 - self.TAU)*w_target, validate_shape=True))
        
        # self.sess.run(tf.global_variables_initializer())
        
        
    
    def predict(self, batch):
        return self.sess.run(self.net, feed_dict={
            self.states: batch[0],
            self.actions:batch[1]
        }) 
    def target_predict(self, batch):
        return self.sess.run(self.target_net, feed_dict={
            self.states: batch[0],
            self.actions:batch[1]
        })
    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })[0]

    def train(self, batch ,target_q):
        self.sess.run(self.optimize, feed_dict={
            self.states: batch[0],
            self.actions:batch[1],
            self.target_q:target_q
        })
        # return self.sess.run(self.loss, feed_dict={
        #     self.states: batch[0],
        #     self.actions:batch[1],
        #     self.target_q:target_q
        # })



    
    def target_train(self):
        
        self.sess.run(self.assigns)
        
    
    def init_input(self):
        with tf.variable_scope('Inputs'):
            self.states = tf.placeholder(shape=[None,29],dtype=tf.float64, name='Sensor_Readings')
            self.actions = tf.placeholder(shape = [None,3],dtype=tf.float64,name='Actions')
            
    
    def init_network(self,name):
        with tf.variable_scope(name):
            
            L1 = tf.layers.dense(inputs=self.states,name='states_input',units=300,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            L2 = tf.layers.dense(inputs=L1,units=600,name='states_dense1',kernel_initializer=tf.initializers.truncated_normal())
            L3 = tf.layers.dense(inputs=self.actions,name='actions_input',units=600,kernel_initializer=tf.initializers.truncated_normal())
            con = tf.concat([L2,L3], axis=1, name="Concating")
            L4 = tf.layers.dense(inputs=con,name='dense_2',units=600,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            out=  tf.layers.dense(inputs=L4,name='out',units=3,kernel_initializer=tf.initializers.truncated_normal())
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return out,weights
            # tf.add_to_collection('out',self.out)

    def init_ops(self):
        # print("hey ",tf.shape(self.net))
        # print("hey ",tf.shape(self.states))
        # print("hey ",tf.shape(self.actions))
        # print("hey ",len(self.weights))
        # print("hey ",self.action_gradient.shape)
        self.target_q = tf.placeholder(tf.float64,[None,3], name='target_q')
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q,self.net))
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        self.action_grads = tf.gradients(self.net, self.actions)  #GRADIENTS for policy update
        
