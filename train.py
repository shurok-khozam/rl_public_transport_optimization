import math
from random import randint, uniform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import custom_env
from dqn import DQN
from octave_adapter import OctaveAdapter
from datetime import datetime
import os

FILES_CONTEXT = "D:/E/Master/Stage/customized code"
FIGURE_COUNTER = 0
RUNNING_TIME = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
TMP_FOLDER_NAME = "tmp_" + RUNNING_TIME
tmp_folder_exists = os.path.exists(TMP_FOLDER_NAME)
if not tmp_folder_exists:
    os.makedirs(TMP_FOLDER_NAME)
TMP_FOLDER = TMP_FOLDER_NAME + "/"
print("All plots will be stored in folder", TMP_FOLDER_NAME)
print("###########################################################################")

def train():
    print("--> Started training function")
    env = custom_env.CustomEnv()
    episodes = env.MAX_EPISODES
    steps = env.MAX_STEPS
    dqn_agent = DQN(env=env)

    statistics_shape = (episodes, steps + 1)
    statistics_nbr_trains = np.zeros(statistics_shape)
    statistics_v_max = np.zeros(statistics_shape)
    statistics_max_dwell = np.zeros(statistics_shape)
    statistics_density_max_opt = np.zeros(statistics_shape)
    statistics_h_out_mean = np.zeros(statistics_shape)
    statistics_I_mean = np.zeros(statistics_shape)
    statistics_A_mean = np.zeros(statistics_shape)
    statistics_mu_mean = np.zeros(statistics_shape)
    statistics_Q_mean = np.zeros(statistics_shape)
    statistics_P_mean = np.zeros(statistics_shape)
    statistics_dwell_mean = np.zeros(statistics_shape)

    octave_adapter = OctaveAdapter(FILES_CONTEXT)
    for episode in range(episodes):
    #for episode in range(1):
        print("----> Starting episode #", episode, "################################################")

        # Random starting state for the beginning each episode
        nbr_trains = randint(env.MIN_NBR_TRAINS, env.MAX_NBR_TRAINS)
        v_max = round(uniform(env.MIN_V_MAX, env.MAX_V_MAX), 2)
        max_dwell = round(uniform(env.MIN_MAX_DWELL, env.MAX_MAX_DWELL), 2)
        density_max_opt = round(uniform(env.MIN_DENSITY_MAX_OPT, env.MAX_DENSITY_MAX_OPT), 2)
        octave_adapter.write_to_octave(nbr_trains, v_max, max_dwell, density_max_opt)
        octave_adapter.run_octave()
        current_state = octave_adapter.read_from_octave()

        # Statistics arrays
        update_statistics(episode, 0, statistics_nbr_trains, statistics_v_max, statistics_max_dwell,
                          statistics_density_max_opt, statistics_h_out_mean, statistics_I_mean,
                          statistics_A_mean, statistics_mu_mean, statistics_Q_mean, statistics_P_mean,
                          statistics_dwell_mean, nbr_trains, v_max, max_dwell, density_max_opt, current_state)

        for step in range(steps):
            print("----> Starting step #", step, "################################################")
            action = dqn_agent.action(env.convert_to_dqn_array(current_state))

            new_state, new_params, reward, done, done_reason = env.step(octave_adapter, current_state, nbr_trains, v_max, max_dwell, density_max_opt, action)
            dqn_agent.remember(env.convert_to_dqn_array(current_state), action,
                               reward, env.convert_to_dqn_array(new_state), done)
            if dqn_agent.replay():  # internally iterates default (prediction) model
                dqn_agent.target_train()  # iterates target model

            current_state = new_state
            nbr_trains = new_params['nbr_trains']
            v_max = new_params['v_max']
            max_dwell = new_params['max_dwell']
            density_max_opt = new_params['density_max_opt']

            update_statistics(episode, step + 1, statistics_nbr_trains, statistics_v_max, statistics_max_dwell,
                          statistics_density_max_opt, statistics_h_out_mean, statistics_I_mean,
                          statistics_A_mean, statistics_mu_mean, statistics_Q_mean, statistics_P_mean,
                          statistics_dwell_mean, nbr_trains, v_max, max_dwell, density_max_opt, current_state)
            print("<---- Ending step #", step)

        print("<---- Ending episode #", episode)
        
        plot_episode_statistics(episode, steps, statistics_nbr_trains, statistics_v_max, statistics_max_dwell,
                                statistics_density_max_opt, statistics_h_out_mean, statistics_I_mean,
                                statistics_A_mean, statistics_mu_mean, statistics_Q_mean, statistics_P_mean,
                                statistics_dwell_mean)

    dqn_agent.save_model('model')
    print("<-- Ended training function")
    plot_full_statistics(episodes, steps, statistics_nbr_trains, statistics_v_max, statistics_max_dwell,
                         statistics_density_max_opt, statistics_h_out_mean, statistics_I_mean,
                         statistics_A_mean, statistics_mu_mean, statistics_Q_mean, statistics_P_mean,
                         statistics_dwell_mean)

def update_statistics(episode, column, statistics_nbr_trains, statistics_v_max, statistics_max_dwell, statistics_density_max_opt,
                      statistics_h_out_mean, statistics_I_mean, statistics_A_mean, statistics_mu_mean, statistics_Q_mean,
                      statistics_P_mean, statistics_dwell_mean,
                      nbr_trains, v_max, max_dwell, density_max_opt, current_state):
    statistics_nbr_trains[episode, column] = nbr_trains
    statistics_v_max[episode, column] = v_max
    statistics_max_dwell[episode, column] = max_dwell
    statistics_density_max_opt[episode, column] = density_max_opt
    statistics_h_out_mean[episode, column] = current_state['h_out_mean']
    statistics_I_mean[episode, column] = current_state['I_mean']
    statistics_A_mean[episode, column] = current_state['A_mean']
    statistics_mu_mean[episode, column] = current_state['mu_mean']
    statistics_Q_mean[episode, column] = current_state['Q_mean']
    statistics_P_mean[episode, column] = current_state['P_mean']
    statistics_dwell_mean[episode, column] = current_state['dwell_mean']

def plot_episode_statistics(episode, steps, statistics_nbr_trains, statistics_v_max,
                            statistics_max_dwell, statistics_density_max_opt, statistics_h_out_mean,
                            statistics_I_mean, statistics_A_mean, statistics_mu_mean, statistics_Q_mean,
                            statistics_P_mean, statistics_dwell_mean):
    single_plot_2D(episode, steps, get_figure_id(), statistics_nbr_trains, "NBR trains", 'NBR trains Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_v_max, "V Max", 'V Max Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_max_dwell, "Max dwell", 'Max dwell Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_density_max_opt, "Density max Opt", 'Density max Opt Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_h_out_mean, "H_out mean", 'Headway mean Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_I_mean, "I_mean", 'Nbr of pass. in st. Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_A_mean, "A_mean", 'Flow of pass. willing to enter Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_mu_mean, "mu_mean", 'Flow of pass. bearding on Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_Q_mean, "Q_mean", 'Nbr of pass. waiting Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_P_mean, "P_mean", 'Nbr of pass. on the train Statistics')
    single_plot_2D(episode, steps, get_figure_id(), statistics_dwell_mean, "dwell_mean", 'Dwell time (s) Statistics')

def single_plot_2D(episode, steps, figure, statistics_array, label, title):
    x_axis = list(range(steps + 1))
    fig = plt.figure(figure)
    plt.plot(x_axis, statistics_array[episode, :].T, label=label)
    plt.xlabel('Steps')
    plt.ylabel(label)
    plt.title(str(episode) + ': ' + title)
    plt.legend()
    fig.savefig(TMP_FOLDER + str(figure) + "_" + label + ".png")
    plt.show()

def plot_full_statistics(episodes, steps, statistics_nbr_trains, statistics_v_max, statistics_max_dwell,
                         statistics_density_max_opt, statistics_h_out_mean, statistics_I_mean,
                         statistics_A_mean, statistics_mu_mean, statistics_Q_mean, statistics_P_mean,
                         statistics_dwell_mean):
    single_plot_3D(episodes, steps, get_figure_id(), statistics_nbr_trains, "NBR trains", 'NBR trains Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_v_max, "V Max", 'V Max Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_max_dwell, "Max dwell", 'Max dwell Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_density_max_opt, "Density max Opt", 'Density max Opt Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_h_out_mean, "H_out mean", 'Headway mean Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_I_mean, "I_mean", 'Nbr of pass. in st. Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_A_mean, "A_mean", 'Flow of pass. willing to enter Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_mu_mean, "mu_mean", 'Flow of pass. bearding on Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_Q_mean, "Q_mean", 'Nbr of pass. waiting Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_P_mean, "P_mean", 'Nbr of pass. on the train Statistics')
    single_plot_3D(episodes, steps, get_figure_id(), statistics_dwell_mean, "dwell_mean", 'Dwell time (s) Statistics')

def single_plot_3D(episodes, steps, figure, statistics_array, label, title):
    x_axis = range(episodes)
    y_axis = range(steps + 1)
    X, Y = np.meshgrid(x_axis, y_axis)
    fig = plt.figure(figure)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X.T, Y.T, statistics_array, rstride=1, cstride=1,
                cmap=cm.coolwarm, edgecolor='none')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(TMP_FOLDER + str(figure) + "_" + label + ".png")
    plt.show()

def get_figure_id():
    global FIGURE_COUNTER
    figure_id = FIGURE_COUNTER
    FIGURE_COUNTER = FIGURE_COUNTER + 1
    return figure_id

if __name__ == "__main__":
    train()
