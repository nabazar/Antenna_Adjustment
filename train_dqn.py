from dqn_ltes import LtesEnv, DeepQNetwork
import numpy as np
#from pyltes.network import CellularNetwork


LEARNING_RATE = .001
DISCOUNT_RATE = .9
BATCH_SIZE = 12
EPISODES = 1000



lteenv = LtesEnv()
#network = CellularNetwork()
#network.Printer.drawNetwork(fillMethod="Sectors", filename="36AntSectorsMap")


#lteenv.network.Printer.drawNetwork(fillMethod="SINR", filename="36AntSinrMap")
nS = lteenv.numUE #This is only 4
nA = lteenv.action_no #Actions

# lteenv, alpha, gamma, epsilon, epsilon_min, epsilon_decay
dqn = DeepQNetwork(lteenv, LEARNING_RATE, DISCOUNT_RATE, 1, 0.001, 0.995)

#Training
rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit
#TEST_Episodes = 0

for e in range(EPISODES):
    state = lteenv.reset()
    lteenv.network.Printer.drawNetwork(fillMethod="SINR", filename="36AntSinrMap")
    # print(state)
    # state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for time in range(30): #200 is when you "solve" the game. This can continue forever as far as I know
        action = dqn.action_dqn(state)
        nstate, reward, done = lteenv.step(action)
        # nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
        state = nstate
        #done: CartPole fell.
        #time == 209: CartPole stayed upright
        if done or time == 29:
            rewards.append(tot_rewards)
            epsilons.append(dqn.epsilon)
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e+1, EPISODES, tot_rewards, dqn.epsilon))
            break
    #Experience Replay
    if len(dqn.memory) > BATCH_SIZE:
        dqn.experience_replay(BATCH_SIZE)
    #If our current NN passes we are done
    #I am going to use the last 5 runs
    # if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
    #     #Set the rest of the EPISODES for testing
    #     TEST_Episodes = EPISODES - e
    #     TRAIN_END = e
    #     break

np.savetxt(f'LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, DNN Layers = 16, 8, 4.txt', 
            X=rewards, fmt='%10.4f')

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
# fig=plt.figure()
plt.plot(rewards)
plt.title(f'LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, DNN Layers = 16, 8, 4', fontsize = 8)
plt.xlabel('epochs', fontsize = 8)
plt.ylabel('rewards', fontsize = 8)
plt.show()