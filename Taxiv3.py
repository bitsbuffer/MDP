# %%%
import os
import re

import gym
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import QLearning

from .QLearning import QLearner

np.random.seed(1234)
random.seed(1234)


class OpenAI_MDPToolbox:
    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments.
    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control
    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.
    """

    def __init__(self, openAI_env_name: str, render: bool = False, seed=1234, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class
        :param openAI_env_name: Valid name of an Open AI Gym env
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean
        """
        self.env_name = openAI_env_name

        self.env = gym.make(self.env_name, **kwargs)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env.reset()

        if render:
            self.env.render()

        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR()

    def convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob * self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob


def plot_pi_vi(df_run_stats, gammas, base_name):
    os.makedirs("images", exist_ok=True)
    for i in range(len(df_run_stats)):
        fig = px.line(df_run_stats[i], "Iteration", 'Error')
        fig.write_image("images/" + base_name + str(gammas[i]) + ".png")

    num_pi_iterations = pd.DataFrame({"NumIteration": [stat_df.shape[0] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(num_pi_iterations.iloc[:-1, :], "Gamma", 'NumIteration')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-iteration.png")

    mean_val = pd.DataFrame(
        {"Mean V": [stat_df.loc[stat_df.index[-1], 'Mean V'] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(mean_val.iloc[:-1, :], "Gamma", 'Mean V')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-mean-v.png")

    error = pd.DataFrame(
        {"Error": [stat_df.loc[stat_df.index[-1], 'Error'] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(error.iloc[:-1, :], "Gamma", 'Error')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-error.png")


def plot_policy_comparison(pi, vi, q):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(1, 500, 500), y=pi.policy, name="PI", mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(1, 500, 500), y=vi.policy, name="VI", mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(1, 500, 500), y=q.policy, name="Q-learning", mode="lines"))
    fig.update_layout(
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    fig.write_image("./images/policy.png")


def plot_q_learning_error(qs):
    iterations = np.linspace(1, 10000, 100)
    fig = go.Figure()
    for q in qs:
        fig.add_trace(go.Scatter(x=iterations, y=q.error_mean, name=f"lambda = {q.gamma}", mode="lines"))
    fig.update_xaxes(title="Iterations")
    fig.update_yaxes(title="Mean Error")
    fig.write_image("./images/q-learning-error.png")


def run_pi(P, R, gammas):
    pi_run_stats = []
    pis = []
    for gamma in gammas:
        pi = PolicyIteration(P, R, gamma, policy0=None,
                             max_iter=1000, eval_type=1, skip_check=False,
                             run_stat_frequency=1)
        pi.run()
        pis.append(pi)
        pi_run_stats.append(pd.DataFrame(pi.run_stats))
    plot_pi_vi(pi_run_stats, gammas, "pi")
    return pis


def run_vi(P, R, gammas):
    vi_run_stats = []
    vis = []
    for gamma in gammas:
        vi = ValueIteration(P, R, gamma, epsilon=0.001, max_iter=1000, initial_value=0, skip_check=False,
                            run_stat_frequency=1)
        vi.run()
        vis.append(vi)
        vi_run_stats.append(pd.DataFrame(vi.run_stats))
    plot_pi_vi(vi_run_stats, gammas, "vi")
    return vis


def run_q(P, R, gammas):
    qs = []
    for gamma in gammas:
        q = QLearning(P, R, gamma,
                      alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                      epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99,
                      n_iter=10000, skip_check=False, iter_callback=None,
                      run_stat_frequency=1)
        q.run()
        qs.append(q)
    return qs


def compare_policies(env, policy, num_episodes):
    env.seed(1234)
    env.action_space.seed(1234)
    total_penalties, total_rewards = 0, 0

    for _ in range(num_episodes):
        state = env.reset()
        penalties, rewards = 0, 0
        done = False
        while not done:
            action = policy[state]
            state, reward, done, info = env.step(action)
            if reward == -10:
                penalties += 1
            rewards += reward
        total_penalties += penalties
        total_rewards += rewards

    print(f"Total penalties {total_penalties}")
    print(f"Total rewards {total_rewards}")
    return total_rewards, total_penalties


def plot_q_policy(qs):
    fig = go.Figure()
    states = np.linspace(1, 500, 500)
    for q in qs:
        fig.add_trace(go.Scatter(x=states, y=q.policy, name=f"lambda = {q.gamma}", mode="lines"))
    fig.update_layout(
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    fig.write_image("./images/q-policy.png")


if __name__ == '__main__':

    env = OpenAI_MDPToolbox("Taxi-v3", render=True)
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    P, R = env.P, env.R

    ps = run_pi(P, R, gammas)
    vs = run_vi(P, R, gammas)
    qs = run_q(P, R, [0.7, 0.8, 0.9, 0.95, 0.99])

    total_rewards, total_penalties = [], []
    oenv = gym.make("Taxi-v3")
    for q in qs:
        a, b = compare_policies(oenv, q.policy, 100)
        total_rewards.append(a)
        total_penalties.append(b)
    gammas = [0.7, 0.8, 0.9, 0.95, 0.99]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=gammas, y=total_rewards))
    fig.update_xaxes(title="Gammas")
    fig.update_yaxes(title="Rewards")
    fig.write_image("./images/q-learning-rewards.png", type='category')