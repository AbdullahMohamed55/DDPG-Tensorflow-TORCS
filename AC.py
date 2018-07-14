import tensorflow as tf
import numpy as np
import time
from actor import Actor
from critic import Critic
from OU import OU
from ReplayBuffer import ReplayBuffer
from gym_torcs import TorcsEnv

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    # np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # G = tf.Graph()
    # sess = tf.Session(config=config)
    # tf.reset_default_graph()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    

    actor = Actor(sess, BATCH_SIZE, TAU, LRA)
    critic = Critic(sess, BATCH_SIZE, TAU, LRC)

    sess.run(tf.global_variables_initializer())
    # actor = Actor( BATCH_SIZE, TAU, LRA)
    # critic = Critic( BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    # print("Now we load the weight")
    # try:
    #     actor.model.load_weights("actormodel.h5")
    #     critic.model.load_weights("criticmodel.h5")
    #     actor.target_model.load_weights("actormodel.h5")
    #     critic.target_model.load_weights("criticmodel.h5")
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            # print("hey",s_t.shape)
            a_t_original = actor.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            # print("hey",new_states.shape)
            target_q_values = critic.target_predict([new_states, actor.target_predict(new_states)])  
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                # print("main",states.dtype)
                # print("main",actions.dtype)
                # print("main",y_t.dtype)
                loss+=critic.train([states,actions], y_t) 
                a_for_grad = actor.predict(states)
                grads = critic.gradients(states, a_for_grad)
                # print("grads : ",grads.dtype)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
            if done:
                break
        if np.mod(step,1000):
            print("target_q_values : " ,target_q_values)
        if np.mod(i, 3) == 0:
            if (train_indicator):
                pass
                # saver = tf.train.Saver()
                # saver.save(sess, save_path = 'weights/model.ckpt',global_step=1000)
        #         print("Now we save model")
        #         actor.model.save_weights("actormodel.h5", overwrite=True)
        #         with open("actormodel.json", "w") as outfile:
        #             json.dump(actor.model.to_json(), outfile)

        #         critic.model.save_weights("criticmodel.h5", overwrite=True)
        #         with open("criticmodel.json", "w") as outfile:
        #             json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame(1)


