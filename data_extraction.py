import pandas
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

class subject_data():
    """ This is for subject-by-subject data extraction, only used by the kinematic_data class.
    name: letter of subject
    num_tries: number of videos taken
    data: panda dataframe containing all 19 kin variables across time
    gesture_labels: panda dataframe containing gesture time limit indexes - first dimension: start index, second: end index, third: gesture"""
    def __init__(self,name,num_tries=5):
        self.name = name
        self.num_tries = num_tries

        self.kin_data = []
        self.gesture_labels = []
        for k in range(self.num_tries):
            self.kin_data += [pandas.read_csv("Knot_Tying\Kinematics\AllGestures\Knot_Tying_"+str(self.name)+"00"+str(k+1)+".txt",header=None,sep="     ", lineterminator="\n")]
            self.gesture_labels += [pandas.read_csv("Knot_Tying\Transcriptions\Knot_Tying_"+str(self.name)+"00"+str(k+1)+".txt",header=None,sep=" ", lineterminator="\n")]

    def extract_gesture(self,columns_of_interest,gesture_of_interest):
        """returns a list of matrixes whose first dimension is state column index and second dimension is time index"""
        gesture_data = []
        for try_num in range(self.num_tries):
            for k in range(len(self.gesture_labels[try_num][2])):

                #If we find a time range during which the gesture is executed:
                if self.gesture_labels[try_num][2][k] == gesture_of_interest:

                    #From gesture labels we read where to extract the data
                    start_idx = self.gesture_labels[try_num][0][k]
                    end_idx = self.gesture_labels[try_num][1][k]

                    gesture_data_per_col = []
                    for col in columns_of_interest:
                        gesture_data_per_col += [self.kin_data[try_num][start_idx:end_idx][col].values]

                    gesture_data += [np.vstack(gesture_data_per_col)]
        return gesture_data

class gesture_data():
    """We use this class so that the dictionary of gesture data can be modified as we extract data."""
    def __init__(self,gesture_name):
        self.name = gesture_name
        self.data = []

class Kinematic_data():

    def __init__(self, subj_names=("B","C","D","E","F","G","H","I"), num_tries=(4,5,5,5,5,5,3,4), columns_of_interest=(13,14,15),gestures_of_interest=('G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14','G15')):
        self.subj_names = subj_names
        self.columns_of_interest = columns_of_interest
        self.subj_names = subj_names
        self.num_tries = num_tries
        self.gestures_of_interest = gestures_of_interest

        self.initialize_gesture_data_dict()

        self.extract_gesture_data()

    def initialize_gesture_data_dict(self):
        """This way the gesture data in the self.gesture_data dictionary is easily iterable"""
        self.gesture_data = {}
        for gesture_name in self.gestures_of_interest:
            gesture = gesture_data(gesture_name)
            self.gesture_data[gesture_name] = gesture

    def extract_gesture_data(self):
        """Uses the subject_data class to extract data subject per subject"""
        self.subject_data = []
        for sub_ind in range(len(self.subj_names)):
            name = self.subj_names[sub_ind]
            tries = self.num_tries[sub_ind]

            subject = subject_data(name,tries)

            for gesture in self.gestures_of_interest:
                found_gestures = subject.extract_gesture(self.columns_of_interest,gesture)

                #Append the new gestures to the gesture object in the gesture_data dictionary
                self.gesture_data[gesture].data += found_gestures


if __name__ == "__main__":

    subj_names = ["B", "C", "D", "E", "F", "G", "H", "I"]
    num_tries = [4, 5, 5, 5, 5, 5, 3, 4]
    columns_of_interest = [13, 14, 15]
    gestures_of_interest = ['G12','G13']

    kin_data = Kinematic_data(subj_names,num_tries,columns_of_interest,gestures_of_interest)

    G12 = kin_data.gesture_data['G12']

    embed()

    subject_B = subject_data("B",num_tries=4)
    data_B001 = subject_B.kin_data[0]

    # Extraction of state values
    X = np.vstack((data_B001[86:185][16].values,data_B001[86:185][17].values)) #First index is feature and second is time

    #Parameters
    last_t_index = X.shape[1]
    sampling_fq = 30.0
    dt = 1.0/sampling_fq

    #Initial estimate
    A_hat = np.array([[-1.0,0.0],[0.2,-1.0]])

    #Initial values
    x_t = X[:,0]
    x_t_hat = x_t

    #Storage values for plots
    list_x_hats = [x_t_hat]
    list_errors = [0]

    for t in range(80): #t is a time index, not the real time value

        #Real dynamics
        x_tp1 = X[:,t+1]

        #Simulation with estimated A_hat
        x_tp1_hat = A_hat.dot(x_t)

        #error computation
        error = x_tp1_hat-x_tp1

        #Adaptation law
        A_hat_dot = -1.0*error.dot(np.transpose(x_tp1_hat))

        #Computation of new A_hat using Euler's method
        A_hat += A_hat_dot*dt

        #t += 1
        x_t = x_tp1
        x_t_hat = x_tp1_hat

        #Fill in lists for plots
        list_errors.append(error)
        list_x_hats.append(x_t_hat)

    print(A_hat)

    plt.plot([k for k in range(80)],[np.linalg.norm(list_errors[k]) for k in range(80)])
    plt.show()
