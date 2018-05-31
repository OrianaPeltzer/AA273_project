from data_extraction import Kinematic_data

from IPython import embed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas


if __name__ == "__main__":
    subj_names = ["B", "C", "D", "E", "F", "G", "H", "I"]
    num_tries = [4, 5, 5, 5, 5, 5, 3, 4]
    #columns_of_interest = [51, 52, 53,54,55,56,70,71,72,73,74,75]
    columns_of_interest = [k for k in range(39,77)]
    gestures_of_interest = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14','G15']

    state_dimension = len(columns_of_interest)

    kin_data = Kinematic_data(subj_names, num_tries, columns_of_interest, gestures_of_interest)

    dict = {}
    dict[39] = "Slave left x"
    dict[40] = "Slave left y"
    dict[41] = "Slave left z"
    dict[42] = "Slave left R1"
    dict[43] = "Slave left R2"
    dict[44] = "Slave left R3"
    dict[45] = "Slave left R4"
    dict[46] = "Slave left R5"
    dict[47] = "Slave left R6"
    dict[48] = "Slave left R7"
    dict[49] = "Slave left R8"
    dict[50] = "Slave left R9"
    dict[51] = "Slave left Vx"
    dict[52] = "Slave left Vy"
    dict[53] = "Slave left Vz"
    dict[54] = "Slave left VR1"
    dict[55] = "Slave left VR2"
    dict[56] = "Slave left VR3"
    dict[57] = "Slave left Gripper Angle"
    dict[58] = "Slave right x"
    dict[59] = "Slave right y"
    dict[60] = "Slave right z"
    dict[61] = "Slave right R1"
    dict[62] = "Slave right R2"
    dict[63] = "Slave right R3"
    dict[64] = "Slave right R4"
    dict[65] = "Slave right R5"
    dict[66] = "Slave right R6"
    dict[67] = "Slave right R7"
    dict[68] = "Slave right R8"
    dict[69] = "Slave right R9"
    dict[70] = "Slave right VR1"
    dict[71] = "Slave right VR2"
    dict[72] = "Slave right VR3"
    dict[73] = "Slave right Vx"
    dict[74] = "Slave right Vy"
    dict[75] = "Slave right Vz"
    dict[76] = "Slave right Gripper Angle"
    for gesture_name in gestures_of_interest:
        for state in range(len(columns_of_interest)):
            plt.figure()
            for k in range(len(kin_data.gesture_data[gesture_name].data)):
                plt.plot(kin_data.gesture_data[gesture_name].data[k][state, :])
            title = "Gesture: "+gesture_name+" - State: "+str(state+39)+" - "+dict[state+39]
            filename = gesture_name+"_"+str(state+39)
            plt.title(title)
            plt.xlabel("time*30")
            plt.ylabel("Data for each movement")
            plt.savefig("Data_plots/"+filename)
