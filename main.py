#!/usr/bin/env python

import numpy as np
import gym
import argparse
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

PRECISION = 1 # 2 decimal points for continuous states

class Agent(object):

    learning_rate = 0.1
    training_epochs = 200000
    batch_size = 100
    display_step = 1000
    n_rollouts = 1000
    gamma = 0.8
    train_iter = 100
    model = Sequential()
    model.add(Dense(units=32, input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(Activation('relu'))
    model.add(Dense(units=2))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True))

    def __init__(self, env):
        self.env = env
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)

        # self.q = np.zeros()
        self.policy = {}
        self.q = {}
        self.prev_action = 0.0
        self.prev_obs = 0.0

    def act(self, observation):
        observation = tuple(observation)
        if observation in self.policy.keys():
            return self.policy[observation]
        else:
            # self.policy[observation] = np.argmax(self.model.predict(
            #     np.expand_dims(observation, axis=0),
            #     batch_size = 1))
            self.policy[observation] = self.env.action_space.sample()
            return int(self.policy[observation])


# function from https://github.com/wojzaremba/trpo/blob/master/utils.py
def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    q = {}
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards = [], [], []
        terminated = False
        ob = env.reset()
        ob = np.around(ob, PRECISION)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for j in range(max_pathlength):
            action = agent.act(ob)
            obs.append(ob)
            actions.append(action)
            res = env.step(action)
            ob = res[0]
            ob = np.around(ob, PRECISION)
            rewards.append(res[1])
            # state_action = tuple(ob) + (action,)
            ob = tuple(ob)
            if ob in q and len(q[ob])>0:
                q[ob][action] = q[ob][action] + agent.gamma**j * res[1]
            else:
                q[ob] = []
                q[ob].append(0)
                q[ob].append(0)
                q[ob][action] = agent.gamma**j * res[1]

            if res[2]:
                terminated = True
                env.reset()
                break

        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "rewards": np.array(rewards),
                "actions": np.array(actions),
                "terminated": terminated}
        print np.mean(rewards)
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += len(path["rewards"])
    for i in q.keys():
        # TODO: Catered to cartpole
        q[i][0] = q[i][0]/len(rewards)
        if len(q[i]) > 1:
            q[i][1] = q[i][1]/len(rewards)
        else:
            q[i].append(0)
    return q


def api(agent, env):
    new_pol = {}
    for j in range(agent.train_iter):
        pol = new_pol
        training_set = []
        # TODO: generalize, catered to cartpole
        q = rollout(env, agent, 1000, env.spec.timestep_limit)
        opt_act = {}
        for state in q.keys():
            opt_act[state] = np.argmax(q[state]) # Gives action with max val
            if agent.act(state)!=opt_act[state] and q[state][agent.act(state)] <= q[state][opt_act[state]]:
                training_set.append((state, opt_act[state]))
            else:
                training_set.append((state, opt_act[state]))
        import pprint
        # pprint.pprint(training_set)
        x = [item[0] for item in list(training_set)]
        # print [item[-1] for item in training_set]
        y = [item[-1] for item in training_set]
        y = to_categorical(y, num_classes=2)

        agent.model.train_on_batch(x, y)
        for state in q.keys():
            pred = agent.model.predict(np.expand_dims(list(state), axis=0), batch_size = 1)
            new_pol[state] = np.argmax(pred)
    return pol

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    # parser.add_argument('--render', action='store_true')
    # parser.add_argument("--max_timesteps", type=int)
    # parser.add_argument('--num_rollouts', type=int, default=20,
    #                     help='Number of expert roll outs')
    # args = parser.parse_args()

    env = gym.make('CartPole-v0')
    agent = Agent(env)
    max_steps = env.spec.timestep_limit

    for j in range(1000):
        agent.policy = api(agent, env)

        if j % 50 == 0:
            done = False
            obs = env.reset()
            totalr = 0.
            steps = 0
            while not done:
                env.render()
                action = agent.act(obs)
                action = np.argmax(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
            env.close()
            print('returns', totalr)

    # for i in range(agent.n_rollouts):
    #     rollout(env, agent, 100, max_steps)
    #     print('iter', i)
        # obs = env.reset()
        # print obs
        # done = False


    #     while not done:
    #
    #         action = env.action_space.sample()
    #         obs, r, done, _ = env.step(action)
    #         totalr += r
    #         steps += 1
    #         # if args.render:
    #         env.render()
    #         if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
    #         if steps >= max_steps:
    #             break
    #     returns.append(totalr)
    #
    # print('returns', returns)
    # print('mean return', np.mean(returns))
    # print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
