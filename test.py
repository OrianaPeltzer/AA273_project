import random
from pomegranate import *

random.seed(0)

state1 = State( NormalDistribution(1.0, 1.0), name="n1" )
state2 = State( NormalDistribution(2, 1), name="n2" )
state3 = State( NormalDistribution(3, 1), name="n3" )

model = HiddenMarkovModel( name="ExampleModel" )
model.add_state( state1 )
model.add_state( state2 )
model.add_state( state3 )

model.add_transition( model.start, state1, 0.5 )
model.add_transition( model.start, state2, 0.5 )

model.add_transition( state1, state1, 0.2 )
model.add_transition( state1, state2, 0.2 )
model.add_transition( state1, state3, 0.4 )

model.add_transition( state2, state2, 0.4 )
model.add_transition( state2, state1, 0.4 )

model.add_transition(state3,state1, 1.0)

model.add_transition( state1, model.end, 0.2 )
model.add_transition( state2, model.end, 0.2 )

model.bake()

sequence = model.sample()
print(sequence)