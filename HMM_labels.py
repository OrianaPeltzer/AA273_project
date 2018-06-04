#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 18:30:20 2018

@author: zong
"""

from pomegranate import *
import random
import numpy as np
random.seed(0)
from GMM import *

X = [[12, 16, 2, 5], [7, 8, 3, 5, 8], [2, 5, 7]]
labels = [['s1', 's2', 's3', 's3'], [ 's1', 's1', 's2', 's2'], ['s2', 's2', 's3']]
s1  = State(NormalDistribution(5, 2), name="s1")
s2 = State(NormalDistribution(12,1), name="s2")
s3 = State(NormalDistribution(1,4), name="s3")
model = HiddenMarkovModel(name="Gesture_Classifier_HMM")
model.add_state(s1)
model.add_state(s2)
model.add_state(s3)

model.add_transition( model.start,s1, 0.5 )
model.add_transition( model.start,s2, 0.5 )

    
model.add_transition( s1,s2, 0.2)
model.add_transition( s1, s3, 0.2)
    
model.add_transition( s2, s1, 0.5)
model.add_transition( s2, s3, 0.5)

model.add_transition(s3,s2,0.75)
    
model.add_transition( s1, model.end, 0.1)
model.add_transition( s3 ,model.end, 0.25)

model.bake()


model.fit(X, labels=labels, algorithm='labeled',verbose = True)