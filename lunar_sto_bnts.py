import GPy
from sto_bnts import STO_BNTS
import pickle
import numpy as np
import gym

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

max_iter = 500

noise_scale = 0.1
ensemble_size = 1
configs = {'learning_rate': 1e-3, 'training_steps': 10000, 'noise_scale':noise_scale, 'W_std': 1.5, \
     'b_std': 0.05, 'width':256, 'depth':2, 'activation': 'relu'}

ensemble_method = "ntkgp_param"
# ensemble_method = "ntkgp_lin"
# ensemble_method = "deep_ensemble"

parameterization = "standard"


def heuristic_Controller(s, w):
    # every element of w is bounded in [0, 2]
    angle_targ = s[0] * w[0] + s[2] * w[1]
    if angle_targ > w[2]:
        angle_targ = w[2]
    if angle_targ < -w[2]:
        angle_targ = -w[2]
    hover_targ = w[3] * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
    hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

    if s[6] or s[7]:
        angle_todo = w[8]
        hover_todo = -(s[3]) * w[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        a = 2
    elif angle_todo < -w[11]:
        a = 3
    elif angle_todo > +w[11]:
        a = 1
    return a


N_episodes = 20
def lunar_objective(param):
    w = param * 2

    env = gym.make("LunarLander-v2")
    reward_avg = 0
    for n in range(N_episodes):
        cum_rewards = []
        state = env.reset()
        a = heuristic_Controller(state, w)
        for _ in range(1000):
            state, reward, done, info = env.step(a)
            cum_rewards.append(reward)
            if done:
                break
            a = heuristic_Controller(state, w)
        reward_avg += np.sum(cum_rewards) / N_episodes

    return reward_avg


init_size = 20
run_list = np.arange(5)

for itr in run_list:
    log_file_name = "results_sto_bnts/res_ensemble_size_" + str(ensemble_size) + \
        "_lr_" + str(configs["learning_rate"]) + "_training_steps_" + str(configs["training_steps"]) + \
        "_noise_scale_" + str(configs["noise_scale"]) + \
        "_W_std_" + str(configs["W_std"]) + "_b_std_" + str(configs["b_std"]) + "_width_" + str(configs["width"]) + \
        "_depth_" + str(configs["depth"]) + "_act_" + configs["activation"] + \
        "_iter_" + str(itr) + "_init_" + str(init_size) + ".pkl"

    if ensemble_method == "deep_ensemble":
        log_file_name = log_file_name[:-4] + "_deep_ensemble.pkl"
    elif ensemble_method == "ntkgp_lin":
        log_file_name = log_file_name[:-4] + "_ntkgp_lin.pkl"

    init_file_name = "inits/init_itr_" + str(itr) + "_init_" + str(init_size) + ".p"

    bo_ts = STO_BNTS(f=lunar_objective, pbounds={'x1':(0, 1), 'x2':(0, 1), 'x3':(0, 1), 'x4':(0, 1), 'x5':(0, 1), 'x6':(0, 1), \
                                          'x7':(0, 1), 'x8':(0, 1), 'x9':(0, 1), 'x10':(0, 1), 'x11':(0, 1), 'x12':(0, 1)},\
                log_file=log_file_name, \
                use_init=init_file_name, save_init=False, save_init_file=None, \
                T=max_iter, ensemble_size=ensemble_size, configs=configs, ensemble_method=ensemble_method)
    bo_ts.maximize(n_iter=max_iter, init_points=init_size)
