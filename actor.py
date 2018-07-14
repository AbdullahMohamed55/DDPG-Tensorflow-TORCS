#importing libraries
import tensorflow as tf
import numpy as np
import time

#State and action dimensions
state_dim= 29
action_dim = 3

class Actor(object):


    def __init__(self,sess,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        # AG = tf.Graph()
        # with AG.as_default():
        self.init_input()          #Function to create the sensor readings placeholder
        self.net,self.weights = self.init_network("ActorNetwork") #Creates the evaluate network and returns output actions and the weights used
        self.target_net,self.target_weights = self.init_network("ActorTarget_Network")
        self.init_ops()            #Function containing the learning paradigm, contains the optimization of 
                                   #weights using the gradients from critic network
        
        self.assigns = []        #Changes target weights with the given evaluation weights slowly with TAU
        for w_agent, w_target in zip(self.weights, self.target_weights):     # Loops over all weights and zips them together
            self.assigns.append(tf.assign(w_target, self.TAU * w_agent+ (1 - self.TAU)*w_target, validate_shape=True))   
        
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config,graph = AG)
        self.sess.run(tf.global_variables_initializer())
        
     #Training target by copying weights from evaluation network to target network with learning rate of TAU
    def target_train(self):
        self.sess.run(self.assigns)
        

    
    def init_input(self):
        with tf.variable_scope('Inputs'):
            self.states = tf.placeholder(shape=[None,29],dtype=tf.float64, name='Sensor_readings')
            
    #Outputs actions using states (used for current states)
    def predict(self, states):
        return self.sess.run(self.net, feed_dict={
            self.states: states
        }) 
    #Outputs actions using states (used for next states)
    def target_predict(self, states):
        return self.sess.run(self.target_net, feed_dict={
            self.states: states
        })
    #Optimizes the network weights in the direction of the action gradients from the critic network
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.states: states,
            self.action_gradient: action_grads
        })
    #Network architecture
    def init_network(self,name):
        with tf.variable_scope(name):
            # TODO initializer
            L1 = tf.layers.dense(inputs=self.states,name = 'dense1',units=300,activation=tf.nn.relu)
            L2 = tf.layers.dense(inputs=L1,units=600,name = 'dense2',activation=tf.nn.relu)
            accelerate = tf.layers.dense(L2,1,activation=tf.nn.sigmoid,name="Accelerate",kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))
            brake = tf.layers.dense(L2,1,activation=tf.nn.sigmoid,name="Brake",kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))
            steer = tf.layers.dense(L2,1,activation=tf.nn.tanh,name="Steer",kernel_initializer=tf.random_uniform_initializer(-1e-4,1e-4))
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
