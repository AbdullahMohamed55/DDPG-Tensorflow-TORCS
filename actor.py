import tensorflow as tf
import numpy as np
import time

state_dim= 29
action_dim = 3

class Actor(object):


    def __init__(self,sess,BATCH_SIZE,TAU,LEARNING_RATE):
    # def __init__(self,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # AG = tf.Graph()
        # with AG.as_default():
        self.init_input()
        self.net,self.weights = self.init_network("ActorNetwork")
        self.target_net,self.target_weights = self.init_network("ActorTarget_Network")
        self.init_ops()

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config,graph = AG)
        self.sess.run(tf.global_variables_initializer())
        
        
    def target_train(self):
        assigns = []
        for w_agent, w_target in zip(self.weights, self.target_weights):
            assigns.append(tf.assign(w_target, self.TAU * w_agent+ (1 - self.TAU)*w_target, validate_shape=True))
        self.sess.run(assigns)


    
    def init_input(self):
        with tf.variable_scope('Inputs'):
            self.states = tf.placeholder(shape=[None,29],dtype=tf.float64, name='Sensor_readings')
            
    
    def predict(self, states):
        return self.sess.run(self.net, feed_dict={
            self.states: states
        }) 
    def target_predict(self, states):
        return self.sess.run(self.target_net, feed_dict={
            self.states: states
        })         
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.states: states,
            self.action_gradient: action_grads
        })
    def init_network(self,name):
        with tf.variable_scope(name):
            # TODO initializer
            L1 = tf.layers.dense(inputs=self.states,units=300,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            L2 = tf.layers.dense(inputs=L1,units=600,kernel_initializer=tf.initializers.truncated_normal(),activation=tf.nn.relu)
            accelerate = tf.layers.dense(L2,1,activation=tf.nn.sigmoid,name="Accelerate")
            brake = tf.layers.dense(L2,1,activation=tf.nn.sigmoid,name="Brake")
            steer = tf.layers.dense(L2,1,activation=tf.nn.tanh,name="Steer")
            out= tf.concat([accelerate, brake, steer], axis=1, name="Actions")
            weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
            return out,weights
            # tf.add_to_collection('out',self.out)

    def init_ops(self):
        self.action_gradient = tf.placeholder(tf.float64,[None, action_dim],name='action_gradients')
        # print("hey ",self.net.shape)
        # print("hey ",len(self.weights))
        # print("hey ",self.action_gradient.shape)
        self.params_grad = tf.gradients(self.net, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
