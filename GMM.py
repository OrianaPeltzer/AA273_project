from pomegranate import *
import numpy as np
import pandas
from IPython import embed

class GMM():
    def __init__(self, gesture_name="G1", num_files=19, num_Gaussians=6):
        self.gesture_name = gesture_name
        self.path = "Csv_per_gesture/"+self.gesture_name+"/"
        self.num_files = num_files
        self.num_Gaussians = num_Gaussians

        self.extract_samples() # Creation of self.Samples
        self.fit_model() # Fitting model to the samples, creation of self.model

    def extract_samples(self):
        for i in range(self.num_files):
            filepath = self.path + str(i+1) + ".csv"
            data = pandas.read_csv(filepath, header=None, sep=",")
            data = data.values[1:]
            if i==0:
                self.Samples = data
            else:
                self.Samples = np.hstack((self.Samples,data))
        self.Samples = np.transpose(self.Samples)

    def fit_model(self):
        self.model = GeneralMixtureModel([MultivariateGaussianDistribution(np.random.rand(12,1),np.diag([1,1,1,1,1,1,1,1,1,1,1,1])) for k in range(self.num_Gaussians)])
        print("")
        print(" Training Model "+self.gesture_name)
        print("")
        self.model.fit(self.Samples, verbose=True, stop_threshold=1)

    def get_sample_params(self):
        """Returns number of samples in test dataset and their dimension"""
        num_Samples = np.shape(self.Samples)[0]
        size_Samples = np.shape(self.Samples)[1]
        return  num_Samples, size_Samples

    def evaluate_point(self, point):
        if np.shape(point)[1] != np.shape(self.Samples)[1]:
            raise ValueError("The samples are arrays of size (1,"+str(np.shape(self.Samples)[1])+"). Please give a sample that is the same size")
        else:
            log_prob = self.model.log_probability(point)
            predict = self.model.predict(point)
            return predict, log_prob


if __name__ == "__main__":

    num_Gaussians=6

    GMM_G1 = GMM(gesture_name="G1", num_files=19, num_Gaussians=num_Gaussians)
    GMM_G11 = GMM(gesture_name="G11", num_files=36, num_Gaussians=num_Gaussians)
    GMM_G12 = GMM(gesture_name="G12", num_files=70, num_Gaussians=num_Gaussians)
    GMM_G13 = GMM(gesture_name="G13", num_files=75, num_Gaussians=num_Gaussians)
    GMM_G14 = GMM(gesture_name="G14", num_files=98, num_Gaussians=num_Gaussians)
    GMM_G15 = GMM(gesture_name="G15", num_files=73, num_Gaussians=num_Gaussians)

    embed()