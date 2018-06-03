from pomegranate import *
import random
import numpy as np
random.seed(0)


state1 = State(MultivariateGaussianDistribution(np.ones(3),np.diag([1,1,1])),name="State1")
state2 = State(MultivariateGaussianDistribution(np.ones(3),4*np.diag([1,1,1])),name="State2")

model = HiddenMarkovModel(name="TestModel")
model.add_state(state1)
model.add_state(state2)

model.add_transition( model.start, state1, 0.5 )
model.add_transition( model.start, state2, 0.5 )

model.add_transition( state1, state1, 0.2)
model.add_transition( state1, state2, 0.4)

model.add_transition( state2, state1, 0.2)
model.add_transition( state2, state2, 0.4)

model.add_transition( state1, model.end, 0.2)
model.add_transition( state2, model.end, 0.2)


model.bake()


sequence = []
for k in range(10):
    sequence.append(np.ndarray.tolist(np.random.rand(3)))
print("Sample:")
print(sequence)
print("")

print("Transition Matrix")
print(model.dense_transition_matrix())


model.fit([sequence])

print("Fit model to sequence.")

sequence2 = []
for k in range(10):
    sequence2.append(np.ndarray.tolist(np.random.rand(3)))

next_sample = model.predict(sequence2)

logp, path = model.viterbi(sequence2)
print("Log probability")
print(logp)
print("Path: ")
for idx, state in path[1:-1]:
    print(state.name)
print("Next sample:")
print(sequence2)
print("Predicted next sample:")
print(next_sample)


