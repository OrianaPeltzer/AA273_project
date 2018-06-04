from pomegranate import *
import random
import numpy as np
random.seed(0)
from GMM import *


def get_samples(state_labels, subj_names=["B", "C", "D", "E", "F", "G", "H", "I"], num_tries=[4, 5, 5, 5, 5, 5, 3, 4],columns_of_interest = [51, 52, 53,54,55,56,70,71,72,73,74,75]):
    """Transition from line index to column index.
    G1 G11 G12 G13 G14 G15
    The samples is a list of list elements"""
    samples = []
    labels = []

    dict = {"G1": 0,"G11": 1,
            "G12": 2,"G13": 3,
            "G14": 4,"G15": 5}

    for k in range(len(subj_names)):
        s_name = subj_names[k]
        n_tries = num_tries[k]
        for j in range(n_tries):
            sample_sequence = []
            label_sequence = []
            data = pandas.read_csv("Knot_Tying\Kinematics\AllGestures\Knot_Tying_"+str(s_name)+"00"+str(j+1)+".txt",header=None,sep="     ", lineterminator="\n")
            transcriptions = pandas.read_csv("Knot_Tying\Transcriptions\Knot_Tying_"+str(s_name)+"00"+str(j+1)+".txt",header=None,sep=" ", lineterminator="\n")
            for sequence_num in range(np.shape(transcriptions)[1]):

                start_idx = transcriptions[0][sequence_num]
                end_idx = transcriptions[1][sequence_num]

                gesture_name = str(transcriptions[2][sequence_num])
                label = dict[gesture_name]

                for time_index in range(start_idx, end_idx+1):
                    sample_per_col = []
                    for col in columns_of_interest:
                        sample_per_col += [data[col-1][time_index]]
                    sample = np.hstack(sample_per_col)

                    label_sequence += [label]
                    sample_sequence += [sample]
            labels += [np.hstack(label_sequence)]
            samples += [np.hstack(sample_sequence)]

    labels = np.hstack(labels)
    return labels, samples




    return samples, labels

if __name__ == "__main__":

    num_Gaussians = 6

    GMM_G1 = GMM(gesture_name="G1", num_files=19, num_Gaussians=num_Gaussians)
    GMM_G11 = GMM(gesture_name="G11", num_files=36, num_Gaussians=num_Gaussians)
    GMM_G12 = GMM(gesture_name="G12", num_files=70, num_Gaussians=num_Gaussians)
    GMM_G13 = GMM(gesture_name="G13", num_files=75, num_Gaussians=num_Gaussians)
    GMM_G14 = GMM(gesture_name="G14", num_files=98, num_Gaussians=num_Gaussians)
    GMM_G15 = GMM(gesture_name="G15", num_files=73, num_Gaussians=num_Gaussians)
    
    state_G1  = State(GMM_G1.model, name="State_G1")
    state_G11 = State(GMM_G11.model, name="State_G11")
    state_G12 = State(GMM_G12.model, name="State_G12")
    state_G13 = State(GMM_G13.model, name="State_G13")
    state_G14 = State(GMM_G14.model, name="State_G14")
    state_G15 = State(GMM_G15.model, name="State_G15")
    
    model = HiddenMarkovModel(name="Gesture_Classifier_HMM")
    model.add_state(state_G1)
    model.add_state(state_G11)
    model.add_state(state_G12)
    model.add_state(state_G13)
    model.add_state(state_G14)
    model.add_state(state_G15)
    
    
    model.add_transition( model.start, state_G1, 0.5 )
    model.add_transition( model.start, state_G11, 0.5 )
    
    model.add_transition( state_G1, state_G1, 0.2)
    model.add_transition( state_G1, state_G11, 0.4)
    
    model.add_transition( state_G11, state_G1, 0.2)
    model.add_transition( state_G11, state_G11, 0.4)
    
    model.add_transition( state_G1, model.end, 0.2)
    model.add_transition( state_G11, model.end, 0.2)
    
    
    model.bake()
    

    state_labels = ["State_G1","State_G11","State_G12","State_G13","State_G14","State_G15" ]
    labels, sequence = get_samples(state_labels)

    model.fit([sequence], labels = [labels], algorithm='labeled', verbose=True)

    print("Fit model to sequence.")

    embed()

        # 
        # state1 = State(MultivariateGaussianDistribution(np.ones(3), np.diag([1, 1, 1])), name="State1")
        # state2 = State(MultivariateGaussianDistribution(np.ones(3), 4 * np.diag([1, 1, 1])), name="State2")
        # 
        # model = HiddenMarkovModel(name="TestModel")
        # model.add_state(state1)
        # model.add_state(state2)
        # 
        # model.add_transition(model.start, state1, 0.5)
        # model.add_transition(model.start, state2, 0.5)
        # 
        # model.add_transition(state1, state1, 0.2)
        # model.add_transition(state1, state2, 0.4)
        # 
        # model.add_transition(state2, state1, 0.2)
        # model.add_transition(state2, state2, 0.4)
        # 
        # model.add_transition(state1, model.end, 0.2)
        # model.add_transition(state2, model.end, 0.2)
        # 
        # model.bake()
        # 
        # sequence = []
        # for k in range(10):
        #     sequence.append(np.ndarray.tolist(np.random.rand(3)))
        # print("Sample:")
        # print(sequence)
        # print("")
        # 
        # print("Transition Matrix")
        # print(model.dense_transition_matrix())
        # 
        # model.fit([sequence])
        # 
        # print("Fit model to sequence.")
        # 
        # sequence2 = []
        # for k in range(10):
        #     sequence2.append(np.ndarray.tolist(np.random.rand(3)))
        # 
        # next_sample = model.predict(sequence2)
        # 
        # logp, path = model.viterbi(sequence2)
        # print("Log probability")
        # print(logp)
        # print("Path: ")
        # for idx, state in path[1:-1]:
        #     print(state.name)
        # print("Next sample:")
        # print(sequence2)
        # print("Predicted next sample:")
        # print(next_sample)
        # 
