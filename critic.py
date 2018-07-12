import tensorflow as tf
import numpy as np
import time

state_dim= 29
action_dim = 3

class Critic(object):


    def __init__(self,sess,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        
        tf.reset_default_graph()
        self.sess =  tf.InteractiveSession()

        self.init_input()
        #  , self.weights, self.state =self.x
        self.net,self.weights = self.init_network("Network")

        self.target_net,self.target_weights = self.init_network("Target_Network")
        self.init_ops()
        self.sess.run(tf.initialize_all_variables())
        
    
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
        return self.loss

    
    def target_train(self):
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, self.TAU * w_agent+ (1 - self.TAU)*w_target, validate_shape=True))
        tf.get_default_session().run(assigns)
    
    def init_input(self):
        with tf.variable_scope('Inputs'):
            self.states = tf.placeholder(tf.float64,[None,29], name='Sensor_readings')
            self.actions = tf.placeholder(tf.float64,[None,3],name='Actions')
            
    
    def init_network(self,name):
        with tf.variable_scope(name):
            
            L1 = tf.layers.dense(inputs=self.states,units=300,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            L2 = tf.layers.dense(inputs=L1,units=600,kernel_initializer=tf.initializers.truncated_normal())
            L3 = tf.layers.dense(inputs=self.actions,units=600,kernel_initializer=tf.initializers.truncated_normal())
            con = tf.concat([L2,L3], axis=0, name="Concating")
            L4 = tf.layers.dense(inputs=con,units=600,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            out=  tf.layers.dense(inputs=L4,units=3,kernel_initializer=tf.initializers.truncated_normal())
            weights = tf.trainable_variables()
            return out,weights
            # tf.add_to_collection('out',self.out)

    def init_ops(self):
        # print("hey ",self.net.shape)
        # print("hey ",len(self.weights))
        # print("hey ",self.action_gradient.shape)
        self.target_q = tf.placeholder(tf.float64,[None,3], name='target_q')
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q,self.net))
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        self.action_grads = tf.gradients(self.net, self.actions)  #GRADIENTS for policy update
        
