from pomegranate import *
import random
import numpy as np
random.seed(0)
from GMM import *
import matplotlib.pyplot as plt
from IPython import embed


def get_samples(state_labels, subj_names=["B", "C", "D", "E", "F", "G", "H", "I"], num_tries=[4, 5, 5, 5, 5, 5, 3, 4],columns_of_interest = [51, 52, 53,54,55,56,70,71,72,73,74,75]):
    """Transition from line index to column index.
    G1 G11 G12 G13 G14 G15
    The samples is a list of list elements"""
    samples = []
    labels = []

    dict = {"G1": "g1","G11": "g11",
            "G12": "g12","G13": "g13",
            "G14": "g14","G15": "g15","G1\r": "g1","G11\r": "g11",
            "G12\r": "g12","G13\r": "g13",
            "G14\r": "g14","G15\r": "g15"}

    for k in range(len(subj_names)):
        s_name = subj_names[k]
        n_tries = num_tries[k]
        for j in range(n_tries):
            sample_sequence = []
            label_sequence = []
            data = pandas.read_csv("/home/zong/AA273_project/Knot_Tying/kinematics/AllGestures/Knot_Tying_"+str(s_name)+"00"+str(j+1)+".txt",header=None,sep="     ", lineterminator="\n")
            transcriptions = pandas.read_csv("/home/zong/AA273_project/Knot_Tying/transcriptions/Knot_Tying_"+str(s_name)+"00"+str(j+1)+".txt",header=None,sep=" ", lineterminator="\n")
            for sequence_num in range(np.shape(transcriptions)[0]): #change this to index 1 to get half trained data

                start_idx = transcriptions[0][sequence_num]
                end_idx = transcriptions[1][sequence_num]

                gesture_name = str(transcriptions[2][sequence_num])
                label = dict[gesture_name]

                for time_index in range(start_idx, end_idx):
                    sample_per_col = []
                    for col in columns_of_interest:
                        sample_per_col += [data[col-1][time_index]]
                        
                    sample = np.hstack(sample_per_col)
                    
                    label_sequence += [label]
                    sample_sequence += [sample]
                    
            labels += [label_sequence]
            samples += [np.vstack(sample_sequence)]
            

    return labels, samples

#def online_evaluate(sequence, model):
#    
#    k = model.predict(sequence,algorithm='map')
#    
#    if k == -1:
#        prediction = 0
#    else:
#        prediction = k[-1] 
#       
#    return prediction

def plot_results(labels, prediction,key):
    
    dictionary = {'g1':1, 'g11':2, 'g12':3, 'g13':4, 'g13b':5,'g14':6,'g15':7}
    
    if len(labels) != len(prediction):
        raise ValueError('Prediction and samples are not the same length!')
        
    x = [k/30 for k in range(len(labels))]
    
    if key == True:
        ypred = [dictionary[elt] for elt in prediction]
    else:
        dictionary2 = {0:1, 1:2, 2:3, 3:4, 4:6,5:7}
        ypred = [dictionary2[elt] for elt in prediction]
        #ypred = prediction
        
    ylabel = [dictionary[elt] for elt in labels]
    
    plt.plot(x, ypred, label='Predicted')
    plt.plot(x, ylabel, label='True Gesture')
    plt.legend(loc='best')
    plt.title('HMM Gesture Estimate')
    plt.yticks(np.arange(8), ('None','G1','G11','G12','G13a','G13b','G14','G15'))
    plt.show()


if __name__ == "__main__":

    num_Gaussians = 12

    GMM_G1 = GMM(gesture_name="G1", num_files=19, num_Gaussians=num_Gaussians)
    GMM_G11 = GMM(gesture_name="G11", num_files=36, num_Gaussians=num_Gaussians)
    GMM_G12 = GMM(gesture_name="G12", num_files=70, num_Gaussians=num_Gaussians)
    GMM_G13 = GMM(gesture_name="G13", num_files=75, num_Gaussians=num_Gaussians)
    GMM_G14 = GMM(gesture_name="G14", num_files=98, num_Gaussians=num_Gaussians)
    GMM_G15 = GMM(gesture_name="G15", num_files=73, num_Gaussians=num_Gaussians)
    
    state_G1  = State(GMM_G1.model, name="g1")
    state_G11 = State(GMM_G11.model, name="g11")
    state_G12 = State(GMM_G12.model, name="g12")
    state_G13 = State(GMM_G13.model, name="g13")
    state_G14 = State(GMM_G14.model, name="g14")
    state_G15 = State(GMM_G15.model, name="g15")
    
  
    
    model = HiddenMarkovModel()
    model.add_state(state_G1)
    model.add_state(state_G11)
    model.add_state(state_G12)
    model.add_state(state_G13)
    model.add_state(state_G14)
    model.add_state(state_G15)
    
    
    model.add_transition( model.start, state_G1, 0.5)
    model.add_transition( model.start, state_G12, 0.5 )
    
    model.add_transition( state_G1, state_G1, 0.5)
    model.add_transition( state_G1, state_G12, 0.5)
    
    model.add_transition( state_G12,state_G12,0.75)
    model.add_transition( state_G12, state_G13,0.25)
    
    model.add_transition( state_G13, state_G13, 0.5)
    model.add_transition( state_G13, state_G14, 0.5)
    
    model.add_transition( state_G14, state_G14, 0.75)
    model.add_transition( state_G14, state_G15, 0.25)
    
    model.add_transition( state_G15, state_G15, 0.5)
    model.add_transition( state_G15, state_G12, 0.3)
    model.add_transition( state_G15, state_G11 ,0.2)
    
    model.add_transition( state_G11, state_G11, 0.1)
    model.add_transition( state_G11, state_G12, 0.7)
    model.add_transition( state_G11, model.end, 0.2)

    
    
    model.bake()
    

    state_labels = ["State_G1","State_G11","State_G12","State_G13","State_G14","State_G15" ]
    labels, sequence = get_samples(state_labels)
    
    del sequence[22]
    del labels[22]
    
    model.fit(sequence, labels = labels, algorithm='labeled', verbose=True, inertia = 0.9)
    #model.fit(sequence, labels = labels, algorithm='labeled',verbose=True, stop_threshold=1)
    print("Fit model to sequence.")
    
    # Number of the sequence to test
    #num_test = 7
    num_test = 3
    
    # Evaluating test sequence
    test_sequence = sequence[num_test]
    prediction = model.viterbi(test_sequence)

    
    t_mat = model.dense_transition_matrix()
    
    
    # Creating Prediction and label objects for plots
    #Prediction_test = online_evaluate(test_sequence, model)
    Prediction_test = []
    for i in range(1,len(prediction[1])-1):
        Prediction_test.append(prediction[1][i][1].name)
    
    Labels_test = labels[num_test]
    plot_results(Labels_test, Prediction_test,True)
    
    # This creates the online simulation using the forward algorithm
    Prediction_test2=[]
    k = model.forward(test_sequence)
    for i in range(1,len(prediction[1])-1):
        pred2 = np.argmax(k[i][:])
        Prediction_test2.append(pred2)
    
    plot_results(Labels_test,Prediction_test2,False)
    