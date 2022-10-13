# -*- coding: utf-8 -*-

import numpy as np
import GPy
from helper_funcs_sto_bnts import UtilityFunction, acq_max
import pickle
import itertools
import time

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

import jax.numpy as np_jax
from jax import random as random_jax
from jax import vmap
import functools
from bayesian_ntk.utils import get_toy_data
from bayesian_ntk.models import homoscedastic_model
from bayesian_ntk.train import train_model
from bayesian_ntk.predict import Gaussian
from bayesian_ntk import predict, config, train_utils
import math
from collections import namedtuple

np.random.seed(0)

class STO_BNTS(object):
    def __init__(self, f, pbounds, \
                 log_file=None, verbose=1, \
                 use_init=False, save_init=False, save_init_file=None, \
                 T=50, ensemble_size=10, configs=None, ensemble_method="ntkgp_param"):
        """
        """

        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.configs = configs

        self.T = T

        self.use_init = use_init
        self.save_init = save_init
        self.save_init_file = save_init_file

        self.log_file = log_file        
        self.pbounds = pbounds
        self.incumbent = None
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.i = 0

        self.util = None
        
        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'incumbent_x':[], 'values_batch':[], 'all_pred':[]}

        self.verbose = verbose
        
        
    def init(self, init_points):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            y = self.f(x)

            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init
        
        if self.save_init:
            pickle.dump(init, open(self.save_init_file, "wb"))
        

    def maximize(self, n_iter=25, init_points=5):
        self.util_ts = UtilityFunction()

        if not self.initialized:
            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))

                print("[loaded init: {0}; {1}]".format(init["X"], init["Y"]))

                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)

                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                self.init(init_points)


        ensemble_size = self.ensemble_size
        configs = self.configs

        key = random_jax.PRNGKey(10)
        noise_scale = configs["noise_scale"]
        Data = namedtuple('Data', ['inputs', 'targets'])
        
        train_xs = np_jax.array(self.X)
        train_ys = np_jax.array(self.Y.reshape(-1, 1))

        train_data = Data(inputs = train_xs, targets = train_ys)

        ensemble_key = random_jax.split(key, ensemble_size)
        train_method = self.ensemble_method
        pred_all = []
        selected_batch = []

        for i in range(ensemble_size):
            key_ = ensemble_key[i]
            pred_model, pred_param = train_model(key=key_, train_method=train_method, train_data=train_data, test_data=None, \
                             parameterization='standard', \
                             activation=configs["activation"], W_std=configs["W_std"], \
                             b_std=configs["b_std"], width=configs["width"], depth=configs["depth"], \
                             learning_rate=configs["learning_rate"], \
                             training_steps=configs["training_steps"], noise_scale=configs["noise_scale"])

            x_max = acq_max(ac=self.util_ts.utility, pred_model=pred_model, pred_param=pred_param, bounds=self.bounds)
            selected_batch.append(x_max)

        print("selected_batch: ", selected_batch)

        for i in range(n_iter):
            values_batch = []
            for x in selected_batch:
                y = self.f(x)
                self.res['all']['values'].append(y)
                self.res['all']['params'].append(x)

                self.Y = np.append(self.Y, y)
                self.X = np.vstack((self.X, x.reshape((1, -1))))

                values_batch.append(y)

            self.res['all']['values_batch'].append(np.max(values_batch))

            incumbent_x = self.X[np.argmax(self.Y)]
            self.res['all']['incumbent_x'].append(incumbent_x)


            train_xs = np_jax.array(self.X)
            train_ys = np_jax.array(self.Y.reshape(-1, 1))

            train_data = Data(inputs = train_xs, targets = train_ys)

            key = random_jax.PRNGKey(10)
        
            ensemble_key = random_jax.split(key, ensemble_size)
            train_method = self.ensemble_method
            pred_all = []
            selected_batch = []
            
            for i in range(ensemble_size):
                key_ = ensemble_key[i]
                pred_model, pred_param = train_model(key=key_, train_method=train_method, train_data=train_data, test_data=None, \
                                 parameterization='standard', \
                                 activation=configs["activation"], W_std=configs["W_std"], \
                                 b_std=configs["b_std"], width=configs["width"], depth=configs["depth"], \
                                 learning_rate=configs["learning_rate"], \
                                 training_steps=configs["training_steps"], noise_scale=configs["noise_scale"])

                x_max = acq_max(ac=self.util_ts.utility, pred_model=pred_model, pred_param=pred_param, bounds=self.bounds)
                selected_batch.append(x_max)
            
            print("selected_batch: ", selected_batch)

            print("iter {0} <--------> best obs in batch: {1}".format(self.i+1, np.max(values_batch)))

            self.i += 1

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

