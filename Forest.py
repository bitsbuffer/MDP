import os

import gym
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import QLearning
from hiive.mdptoolbox.example import forest

np.random.seed(1234)
random.seed(1234)


def plot_pi_vi(df_run_stats, gammas, base_name):
    os.makedirs("images", exist_ok=True)
    for i in range(len(df_run_stats)):
        fig = px.line(df_run_stats[i], "Iteration", 'Error')
        fig.write_image("images/" + base_name + str(gammas[i]) + ".png")

    num_pi_iterations = pd.DataFrame({"NumIteration": [stat_df.shape[0] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(num_pi_iterations, "Gamma", 'NumIteration')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-iteration.png")

    mean_val = pd.DataFrame(
        {"Mean V": [stat_df.loc[stat_df.index[-1], 'Mean V'] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(mean_val, "Gamma", 'Mean V')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-mean-v.png")

    error = pd.DataFrame(
        {"Error": [stat_df.loc[stat_df.index[-1], 'Error'] for stat_df in df_run_stats], "Gamma": gammas})
    fig = px.bar(error, "Gamma", 'Error')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/" + base_name + "-error.png")


def plot_policy_comparison(pi, vi):
    fig = go.Figure()
    len_policy = len(pi.policy)
    fig.add_trace(go.Scatter(x=np.linspace(1, len_policy, len_policy), y=pi.policy, name="PI", mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(1, len_policy, len_policy), y=vi.policy, name="VI", mode="lines"))
    fig.update_layout(
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            title="Actions"
        ),
        xaxis=dict(title="States")
    )
    fig.write_image("./images/policy.png")


def plot_policy_time_comparison(ps, vs):
    fig = go.Figure()
    ps_time = [pi.time for pi in ps]
    ps_gammas = [pi.gamma for pi in ps]
    vs_time = [pi.time for pi in vs]
    vs_gammas = [pi.gamma for pi in vs]
    fig.add_trace(go.Bar(x=ps_gammas, y=ps_time, name="PI"))
    fig.add_trace(go.Bar(x=vs_gammas, y=vs_time, name="VI"))
    fig.update_layout(
        yaxis=dict(
            title="Time"
        ),
        xaxis=dict(title="Gamma", type="category")
    )
    fig.write_image("./images/time-gamma.png")

def plot_value_func(pi, vi):
    fig = go.Figure()
    len_policy = len(pi.policy)
    fig.add_trace(go.Scatter(x=np.linspace(1, len_policy, len_policy), y=pi.V, name="PI", mode="lines"))
    fig.add_trace(go.Scatter(x=np.linspace(1, len_policy, len_policy), y=vi.V, name="VI", mode="lines"))
    fig.update_layout(
        yaxis=dict(title="Value Function"),
        xaxis=dict(title="States")
    )
    fig.write_image("./images/value_function.png")


def plot_q_learning_error(qs):
    fig = go.Figure()
    for q in qs:
        iterations = np.linspace(1, q.S, 100)
        fig.add_trace(go.Scatter(x=iterations, y=q.error_mean, name=f"lambda = {q.gamma}", mode="lines"))
    fig.update_xaxes(title="Iterations")
    fig.update_yaxes(title="Mean Error")
    fig.write_image("./images/q-learning-error.png")


def tune_gamma_pi(P, R, gammas):
    pi_run_stats = []
    pis = []
    for gamma in gammas:
        pi = PolicyIteration(P, R, gamma, policy0=None,
                             max_iter=1000, eval_type=1, skip_check=False,
                             run_stat_frequency=1)
        pi.run()
        pis.append(pi)
        pi_run_stats.append(pd.DataFrame(pi.run_stats))
    return pis, pi_run_stats


def tune_gamma_vi(P, R, gammas):
    vi_run_stats = []
    vis = []
    for gamma in gammas:
        vi = ValueIteration(P, R, gamma, epsilon=0.0001, max_iter=1000, initial_value=0, skip_check=False,
                            run_stat_frequency=1)
        vi.run()
        vis.append(vi)
        vi_run_stats.append(pd.DataFrame(vi.run_stats))
    return vis, vi_run_stats


def run_q(P, R, gammas):
    qs = []
    for gamma in gammas:
        q = QLearning(P, R, gamma,
                      alpha=0.1, alpha_decay=0.99, alpha_min=0.001,
                      epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.99,
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
    for q in qs:
        states = np.linspace(1, q.S, q.S)
        fig.add_trace(go.Scatter(x=states, y=q.policy, name=f"lambda = {q.gamma}", mode="lines"))
    fig.update_layout(
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            title="Actions"
        ),
        xaxis=dict(title="States")
    )
    fig.write_image("./images/q-policy.png")


def run_pi(P, R):
    pi = PolicyIteration(P, R, 0.99, policy0=None,
                         max_iter=1000, eval_type=1, skip_check=False,
                         run_stat_frequency=1)
    pi.run()
    pi_run_stats = pd.DataFrame(pi.run_stats)
    return pi, pi_run_stats


def run_vi(P, R):
    vi = ValueIteration(P, R, 0.99, epsilon=0.0001, max_iter=1000, initial_value=0, skip_check=False,
                        run_stat_frequency=1)
    vi.run()
    vi_run_stats = pd.DataFrame(vi.run_stats)
    return vi, vi_run_stats


def compare_states():
    states = [1000, 1200, 1400, 1600, 1800, 2000]
    pis = []
    vis = []
    pi_run_stats = []
    vi_run_stats = []
    for state in states:
        P, R = forest(S=state, p=0.4)
        pi, pi_run_stat = run_pi(P, R)
        vi, vi_run_stat = run_vi(P, R)
        pis.append(pi)
        vis.append(vi)
        pi_run_stats.append(pi_run_stat)
        vi_run_stats.append(vi_run_stat)
    num_pi_iterations = pd.DataFrame({"Num Iteration": [stat_df.shape[0] for stat_df in pi_run_stats], "State": states})

    fig = px.bar(num_pi_iterations, "State", 'Num Iteration')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/pi-state-iteration.png")
    # for vi
    num_vi_iterations = pd.DataFrame({"Num Iteration": [stat_df.shape[0] for stat_df in vi_run_stats], "State": states})
    fig = px.bar(num_vi_iterations, "State", 'Num Iteration')
    fig.update_xaxes(type='category')
    fig.update_traces(marker_color='indianred')
    fig.write_image("images/vi-state-iteration.png")
    return pis, pi_run_stats, vis, vi_run_stats


def find_penalties(qs, optimal_policy):
    penalties = []
    for q in qs:
        penalties.append((optimal_policy != q.policy).sum())
    return penalties


if __name__ == '__main__':

    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    P, R = forest(S=1000, p=0.4)

    ps, pi_run_stats = tune_gamma_pi(P, R, gammas)
    plot_pi_vi(pi_run_stats, gammas, "vi")
    vs, vi_run_stats = tune_gamma_vi(P, R, gammas)
    plot_pi_vi(vi_run_stats, gammas, "pi")
    plot_policy_time_comparison(ps, vs)
    compare_states()

    q_gammas = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    P, R = forest(S=5, p=0.1)
    qs = run_q(P, R, q_gammas)
    plot_q_learning_error(qs)
    plot_q_policy(qs)
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