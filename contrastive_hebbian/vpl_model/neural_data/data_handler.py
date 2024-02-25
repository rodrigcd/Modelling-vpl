import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vpl_model.utils import periodic_kernel


class DataHandler(object):

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_pickle(data_path)
        self.main_areas = ["V1", "LM", "LI"]
        self.main_training_steps = ["naive", 45, 35, 30, 25, 20]

    @staticmethod
    def general_filters(df, by_RF_max=False):
        df = df[df.OTC_S2N >= 1]
        df = df[df.RF_overlap == True]
        if by_RF_max:
            df = df[df.RF_max >= .5]
        else:
            df = df[df.RF_S2N >= 2.0]
        return df

    def get_bandwidth_traininglvl_area(self, make_plot=False):
        training_steps = self.df.training_lvl.unique()
        brain_area = self.df.area_ID_str.unique()
        bandwidths = {}
        print("number of cells per step and area: ")
        for step in training_steps:
            for area in brain_area:
                df = self.general_filters(self.df)
                df = df[(df.training_lvl == step) & (df.area_ID_str == area)]
                bandwidths[(step, area)] = df.bandwidth_deg.values
                if step in self.main_training_steps and area in self.main_areas:
                    print("training lvl", step, "area", area, "n cells", len(bandwidths[(step, area)]))
        if make_plot:
            area_to_plot = self.main_areas
            training_to_plot = self.main_training_steps
            f, ax = plt.subplots(1, len(area_to_plot), figsize=(15, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(training_to_plot)))
            for i, area in enumerate(area_to_plot):
                for j, step in enumerate(training_to_plot):
                    h, bins = np.histogram(bandwidths[(step, area)],
                                           bins=20, density=True)
                    ax[i].plot(bins[:-1], h, color=colors[j], label="Training lvl " + str(step))
                ax[i].set_title(area)
                ax[i].legend()
                ax[i].set_xlabel("Bandwidth (deg)")
                ax[i].set_ylabel("Density")
            plt.show()
        return bandwidths

    def get_orientation_traininglvl_area(self, make_plot=False):
        training_steps = self.df.training_lvl.unique()
        brain_area = self.df.area_ID_str.unique()
        orientations = {}
        print("number of cells per step and area: ")
        for step in training_steps:
            for area in brain_area:
                df = self.general_filters(self.df)
                df = df[(df.training_lvl == step) & (df.area_ID_str == area)]
                orientations[(step, area)] = df.pref_ori_deg.values
                if step in self.main_training_steps and area in self.main_areas:
                    print("training lvl", step, "area", area, "n cells", len(orientations[(step, area)]))
        if make_plot:
            area_to_plot = self.main_areas
            training_to_plot = self.main_training_steps
            f, ax = plt.subplots(1, len(area_to_plot), figsize=(15, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(training_to_plot)))
            for i, area in enumerate(area_to_plot):
                for j, step in enumerate(training_to_plot):
                    h, bins = np.histogram(orientations[(step, area)],
                                           bins=20, density=True)
                    ax[i].plot(bins[:-1], h, color=colors[j], label="Training lvl " + str(step))
                ax[i].set_title(area)
                ax[i].legend()
                ax[i].set_xlabel("Orientation (deg)")
                ax[i].set_ylabel("Density")
            plt.show()
        return orientations

    def angle_bandwidth_generated_weights(self, input_dim, hidden_dim, angles):
        orientations = self.get_orientation_traininglvl_area()
        bandwidths = self.get_bandwidth_traininglvl_area()
        orientations = orientations[("naive", "V1")]
        bandwidths = bandwidths[("naive", "V1")]
        available_cells = len(orientations)
        sorted_ori_index = np.argsort(orientations)
        sorted_ori = orientations[sorted_ori_index]
        sorted_bandwidths = bandwidths[sorted_ori_index]
        skip_index = np.arange(0, available_cells, available_cells//hidden_dim)
        sampled_ori = sorted_ori[skip_index]
        sampled_bandwidths = sorted_bandwidths[skip_index]
        #np.random.shuffle(sampled_bandwidths)
        sigma = sampled_bandwidths/2.35482
        W = np.zeros((hidden_dim, input_dim))
        for i in range(hidden_dim):
            mu = sampled_ori[i]
            W[i, :] = periodic_kernel(angles, mu, sigma=sigma[i], period=180)
        return W, sampled_ori, sigma


if __name__ == "__main__":
    data_path = "../../../../tmp_data/20240225031102_orituneData.pkl"
    data_handler = DataHandler(data_path)

    input_dim = 180
    angles = np.linspace(0, 180, input_dim)
    hidden_dim = 100
    W, sampled_ori, sampled_bandwidths = data_handler.angle_bandwidth_generated_weights(input_dim, hidden_dim, angles)
    print("W shape", W.shape)
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    bins = 20
    ax[0].hist(sampled_ori, bins=bins)
    ax[0].set_title("Sampled orientation")
    ax[1].hist(sampled_bandwidths*2.35482, bins=bins)
    ax[1].set_title("Sampled bandwidth")
    plt.show()
    print("Data loaded from", data_handler.data_path)