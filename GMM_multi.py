from pomegranate import *
import numpy as np
import pandas
from IPython import embed

Samples = []


model = GeneralMixtureModel([MultivariateGaussianDistribution(np.random.rand(3,1),np.diag([1,1,1])) for k in range(6)])

Samples = np.random.rand(25,3)

model.fit(Samples, verbose=True)

print(Samples)

