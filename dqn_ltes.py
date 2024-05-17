import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
from pyltes.network import CellularNetwork

from keras import layers
# from tensorflow.keras import layers

import math

random.seed(10)
np.random.seed(0)
tf.random.set_seed(1)


class LtesEnv():
    def __init__(self, numberOfBS=36, radius=1666, powerBS=40, numUE=100,
                 environment='Free-Space', ):

        self.environment = environment
        self.numUE = numUE
        self.action_no = numberOfBS * (6 ** 4)  ### numberOfBS * sectroinBS * actionforsector
        self.network = CellularNetwork()
        self.network.Generator.createHoneycombBSdeployment(radius, numberOfBS=numberOfBS)
        self.network.setPowerInAllBS(powerBS)
        self.network.Generator.insertUErandomly(numUE)
        plt.show()
        # self.network.Printer.drawNetwork(fillMethod="SINR", filename="36AntSinrMap")
        self.Input_sector_model = 5


        ### action_per_parameter
        self.elec_tilte_action = [0, 3, 6, 9, 12, 15]
        self.meca_tilte_action = [0, 3, 6, 9, 12, 15]##########################
        self.vert_beamwidth_action = [4.4, 6.8, 9.4 , 10, 13.5]
        self.hori_beamwidth_action = [45, 55, 65, 70, 75, 85]

        self.total_action = [self.elec_tilte_action, self.meca_tilte_action,
                             self.vert_beamwidth_action,
                             self.hori_beamwidth_action]

        if numberOfBS % 3 == 1:
            print("Incorrect number of BaseStations for sector antennas. Increasing the number.")
        numberOfBS = math.ceil(numberOfBS / 3.0)
        self.numberOfBS = 1 # numberOfBS*3



    # def rewards(self, ):
    #     reward = np.sum(np.asarray(self.network.SinrForUEsVec))
    #     return reward

    def step(self, action):
        sum_sinr, tilt_elec, tilt_mech, HPBW_v, HPBW_h = [], [], [], [], []
        state = np.zeros((self.numberOfBS, self.Input_sector_model))
        for i in range(self.numberOfBS):
            tilt_elec.append(action[i][0])
            tilt_mech.append(action[i][1])
            HPBW_v.append(action[i][2])
            HPBW_h.append(action[i][3])

        self.network.connectUsersToTheBestBS(environment=self.environment,
                                             tilt_elec=tilt_elec,
                                             tilt_mech=tilt_mech,
                                             HPBW_v=HPBW_v,
                                             HPBW_h=HPBW_h)

        for i in range(self.numberOfBS):
            us_per_bs = np.where(np.array(self.network.BSForUEVec) == i)[0][: self.Input_sector_model]
            state[i, :] = [self.network.SinrForUEsVec[k] for k in us_per_bs]

        # sum_sinr.append()
        reward = sum(self.network.SinrForUEsVec)
        done = False

        return state, reward, done

    def reset(self,):

        sum_sinr, tilt_elec, tilt_mech, HPBW_v, HPBW_h = [], [], [], [], []
        self.action = np.zeros((self.numberOfBS, 4))

        for i in range(self.numberOfBS):
            sub_action = [np.random.choice(self.elec_tilte_action,1),
                          np.random.choice(self.meca_tilte_action,1),
                          np.random.choice(self.vert_beamwidth_action,1),
                          np.random.choice(self.hori_beamwidth_action,1),]

            self.action[i] = sub_action

        state = np.zeros((self.numberOfBS, self.Input_sector_model))
        for i in range(self.numberOfBS):
            tilt_elec.append(self.action[i][0])
            tilt_mech.append(self.action[i][1])
            HPBW_v.append(self.action[i][2])
            HPBW_h.append(self.action[i][3])


        self.network.connectUsersToTheBestBS(environment=self.environment,
                                             tilt_elec=tilt_elec,
                                             tilt_mech=tilt_mech,
                                             HPBW_v=HPBW_v,
                                             HPBW_h=HPBW_h)

        for i in range(self.numberOfBS):
            us_per_bs = np.where(np.array(self.network.BSForUEVec) == i)[0][: self.Input_sector_model]
            # a =
            state[i, :] = [self.network.SinrForUEsVec[i] for i in us_per_bs]

        # f
        # print("ss")
        return state



class DeepQNetwork:
    def __init__(self, lteenv, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.lteenv = lteenv
        self.nS = self.lteenv.Input_sector_model
        self.nA = 4
        self.memory = deque([], maxlen=4000)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = [self.sector_model() for _ in range(self.lteenv.numberOfBS)]
        self.loss = {}
        for i_sector in range(self.lteenv.numberOfBS):
            self.loss[i_sector] = []
        # self.action = [np.zeros(4) for _ in range(self.lteenv.numberOfBS)]
        self.action = np.zeros((self.lteenv.numberOfBS, len(self.lteenv.total_action), 1))



    def sector_model(self):
        inputs = keras.Input(shape=(self.lteenv.Input_sector_model,))

        x = layers.Dense(16, activation="relu")(inputs)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(4, activation="relu")(x)

        elec_tilte = layers.Dense(len(self.lteenv.elec_tilte_action))(x)
        meca_tilte = layers.Dense(len(self.lteenv.meca_tilte_action))(x)
        vert_beamwidth  = layers.Dense(len(self.lteenv.vert_beamwidth_action))(x)
        hori_beamwidth = layers.Dense(len(self.lteenv.hori_beamwidth_action))(x)

        model_per_sector = keras.Model(inputs=inputs,
                                       outputs=[elec_tilte, meca_tilte, vert_beamwidth, hori_beamwidth],
                                       name="sector_model")
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model_per_sector.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(
                          lr=self.alpha))
        return model_per_sector

    def action_dqn(self, state):
        for i in range(self.lteenv.numberOfBS):
            if np.random.rand() <= self.epsilon:
                sub_action = [np.random.choice(self.lteenv.elec_tilte_action,1),
                              np.random.choice(self.lteenv.meca_tilte_action,1),
                              np.random.choice(self.lteenv.vert_beamwidth_action,1),
                              np.random.choice(self.lteenv.hori_beamwidth_action,1),]

            # return random.randrange(self.nA)  # Explore
                self.action[i] = sub_action
                # print('random', end='')

            else:
                state_reshape = np.array(state[i,:]).reshape(1, self.nS)
                action_vals = self.model[i].predict(state_reshape) ##[elec, mec, v, h] # Exploit: Use the NN to predict the correct action from this state
                for idx_action, sub_action in enumerate(self.lteenv.total_action):

                    self.action[i,idx_action] = sub_action[int(np.argmax(action_vals[idx_action]))]

        return self.action

    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        # Execute the experience replay
        minibatch = random.sample(self.memory, batch_size)  # Randomly sample from memory
        # state, action, reward, nstate, done
        for index_sector, sector_model in enumerate(self.model):

            # Convert to numpy for speed by vectorization
            # for idx_model in range(self.lteenv.numberOfBS):
            # x = []
            # y = []
            # np_array = np.array(minibatch)
            # st = np.zeros((0, self.nS))  # States
            # nst = np.zeros((0, self.nS))  # Next States
            st, nst = [], []
            for i in range(len(minibatch)):  # Creating the state and next state np arrays
                np_array = np.array(minibatch[i])
                st_mini = np.array(np_array[0][index_sector])
                nst_mini = np.array(np_array[3][index_sector])
                # st = np.append(st, st_mini, axis=0)
                # nst = np.append(nst, nst_mini, axis=0)
                st.append(st_mini)
                nst.append(nst_mini)
            st = np.array(st)
            nst = np.array(nst)
            st_predict = sector_model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
            nst_predict = sector_model.predict(nst)
            index_batch = 0
            amax_action = np.array([np.amax(out, axis=1) for out in nst_predict])
            for _, action, reward, _, done in minibatch:
                # x.append(state)
                # x.append(np.array(np_array[0][index_sector]))
                # Predict from state
                # nst_action_predict_model = nst_predict[index]
                if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                    target = reward*np.ones((4, batch_size))
                else:  # Non terminal
                    # amax_action = np.array([np.amax(out, axis=1) for out in nst_predict])
                    target = reward + self.gamma * amax_action[:,index_batch] # 4
                # target_f = st_predict[index_batch]
                # # 4 * (batch_size* action no.)
                # target_f[action] = target

                for idx_action, sub_action in enumerate(self.lteenv.total_action):
                    # print(action[index_sector])
                    a = sub_action.index(action[index_sector][idx_action])
                    c = st_predict[idx_action]
                    d = st_predict[idx_action][index_batch]
                    b = st_predict[idx_action][index_batch, sub_action.index(action[index_sector][idx_action])]
                    st_predict[idx_action][index_batch, sub_action.index(action[index_sector][idx_action])] = target[idx_action]


                # target_f = [
                # y.append(target_f)
                index_batch += 1
            # Reshape for Keras Fit
            x_reshape = np.array(st).reshape(batch_size, self.nS)
            y_reshape = st_predict.copy()
            epoch_count = 1  # Epochs is the number or iterations
            hist = sector_model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
            # Graph Losses


            self.loss[index_sector].append(hist.history)
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
