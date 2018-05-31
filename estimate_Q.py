from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import pandas


def initialize_As(state_dimension, gestures_of_interest):
    """Beware: This function initializes all A_gesture.npy matrixes that will be optimized using regression,
    and overwrites any already existing As. Make sure not to overwrite for nothing"""
    A = np.array([[-3,0.1],[0.0,-1.0]])
    for gesture_name in gestures_of_interest:
        np.save("A_"+gesture_name, A)
    return

def perform_regression(initial_A,gesture_data,plotting=0, sampling_fq = 30):
    """This function returns a new A after going through the gesture_data.
    Initial_A must be a square matrix with as dimension the number of states corresponding to gesture_data.
    Gesture data is a list of gesture samples, each sample being an array of dimensions (columns of state, time index)
    if plotting=1, then error on state estimates is plotted
    Sampling_fq is in Hz and corresponds to the data acquisition frequency"""

    # Initial estimate
    A_hat = initial_A

    # Storage values for plots
    list_errors = [0]

    dt = 1.0 / sampling_fq

    for X in gesture_data:

        # Parameters
        last_t_index = X.shape[1]

        # Initial values
        x_t = X[:, 0]
        x_t_hat = x_t

        for t in range(last_t_index-1):  # t is a time index, not the real time value

            # Real dynamics
            x_tp1 = X[:, t + 1]

            # Simulation with estimated A_hat
            x_tp1_hat = A_hat.dot(x_t)

            # error computation
            error = x_tp1_hat - x_tp1

            # Adaptation law
            A_hat_dot = -1.0 * error.dot(np.transpose(x_tp1_hat))

            # Computation of new A_hat using Euler's method
            A_hat += A_hat_dot * dt

            # t += 1
            x_t = x_tp1
            x_t_hat = x_tp1_hat

            # Fill in lists for plots
            list_errors.append(np.linalg.norm(error))

        print(A_hat)

    if plotting==1:
        plt.plot(list_errors)
        plt.show()

    return A_hat

def linear_regression(initial_A,gesture_data,plotting=0, sampling_fq = 30):
    return


if __name__ == "__main__":
    subj_names = ["B", "C", "D", "E", "F", "G", "H", "I"]
    num_tries = [4, 5, 5, 5, 5, 5, 3, 4]
    columns_of_interest = [51, 52, 53,54,55,56,70,71,72,73,74,75]
    gestures_of_interest = ['G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12','G13','G14','G15']

    state_dimension = len(columns_of_interest)

    kin_data = Kinematic_data(subj_names, num_tries, columns_of_interest, gestures_of_interest)

    #G12 = kin_data.gesture_data['G12'].data

    # Only uncomment this step if you want to restart from scratch
    print("Initializing As")
    initialize_As(state_dimension, gestures_of_interest)
    print("As initialized and stored!")
    # ------------------------------------------------------------ #

    for gesture_name in gestures_of_interest:



