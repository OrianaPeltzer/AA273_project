#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 23:57:47 2018

@author: zong
"""
from pomegranate import *
import random
import numpy as np
random.seed(0)

dim_feature = 12
n_mix = 3

model = HiddenMarkovModel()
mixtures_0 = [MultivariateGaussianDistribution(np.random.randn(dim_feature) + i, np.identity(dim_feature)) for i in range(n_mix)]
mixtures_1 = [MultivariateGaussianDistribution(np.random.randn(dim_feature) + i, np.identity(dim_feature)) for i in range(n_mix)]
mixtures_2 = [MultivariateGaussianDistribution(np.random.randn(dim_feature) + i, np.identity(dim_feature)) for i in range(n_mix)]
state_0 = State(GeneralMixtureModel(mixtures_0), name="s1")
state_1 = State(GeneralMixtureModel(mixtures_1), name="s2")
state_2 = State(GeneralMixtureModel(mixtures_2), name="s3")

model.add_state(state_0)
model.add_state(state_1)
model.add_state(state_2)

model.add_transition(model.start, state_0, 1.0)
model.add_transition(state_0, state_0, 0.5)
model.add_transition(state_0, state_1, 0.5)
model.add_transition(state_1, state_1, 0.5)
model.add_transition(state_1, state_2, 0.5)
model.add_transition(state_2, state_2, 0.5)
model.add_transition(state_2, model.end, 0.5)

model.bake()

x1 = np.random.normal(5, 1, (4,dim_feature))
x2 = np.random.normal(10, 1, (4,dim_feature))
x3 = np.random.normal(2, 1, (3,dim_feature))
'''x1 = x1.transpose()
x2 = x2.transpose()
x3 = x3.transpose()'''

X = [x1, x2, x3]
labels = [['s1', 's2', 's3', 's3'], [ 's1', 's1', 's2', 's2'], ['s2', 's2', 's3']]

model.fit(X, labels=labels, algorithm='labeled',verbose = True)