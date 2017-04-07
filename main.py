#!/usr/bin/env python

import numpy as np
import gym
import argparse
import gym

PRECISION = 2 # 2 decimal points for continuous states

class Agent(object):

    learning_rate = 0.01
    training_epochs = 200000
    batch_size = 100
    display_step = 1000
    n_rollouts = 1000

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
            # First time encountering this observation, panic
            self.policy[observation] = self.env.action_space.sample()
            return self.policy[observation]



# function from https://github.com/wojzaremba/trpo/blob/master/utils.py
def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        obs, actions, rewards = [], [], []
        terminated = False
        ob = env.reset()
        ob = np.around(ob, PRECISION)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for _ in range(max_pathlength):
            action = agent.act(ob)
            ob = np.around(ob, PRECISION)
            obs.append(ob)
            actions.append(action)
            res = env.step(action)
            ob = res[0]
            rewards.append(res[1])

            # if obs in q:
            #     q[obs] = q[obs] + gamma**j * r
            # else:
            #     q[obs] = gamma**j * r

            if res[2]:
                terminated = True
                env.reset()
                break
        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "rewards": np.array(rewards),
                "actions": np.array(actions),
                "terminated": terminated}
        paths.append(path)
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        timesteps_sofar += len(path["rewards"])
    return paths

# Computes Q state value function from rollouts
def get_q(paths, gamma):
    q = {}
    K = len(paths)
    for i in range(K):
        obs = paths[i]["obs"]
        rewards = paths[i]["rewards"]
        actions = paths[i]["rewards"]

        path_len = len(r)
        for j in range(path_len):
            ob = obs[j]
            r = rewards[j]
            act = actions[j]

            if obs in q:
                q[obs] = q[obs] + gamma**j * r
            else:
                q[obs] = gamma**j * r

    # Now average each q array



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

    returns = []
    observations = []
    actions = []
    for i in range(agent.n_rollouts):
        rollout(env, agent, 100, max_steps)
        print('iter', i)
        # obs = env.reset()
        # print obs
        # done = False
        # totalr = 0.
        # steps = 0

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
