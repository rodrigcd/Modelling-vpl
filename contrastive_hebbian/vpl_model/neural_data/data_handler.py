import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vpl_model.utils import periodic_kernel, lowess
from vismap import config # settings such as OTC_S2N_min and BW_min are in here
from vismap.datatools import data_loader


class DataHandler(object):

    def __init__(self, data_path):
        self.data_path = data_path
        #self.df = pd.read_pickle(data_path)
        allCells, _ = data_loader(len_scale='2_10', filter_low_bw=True, filter_S2N=True,
                                  filter_TL_only=True, loc="local")
        self.df = allCells.allCells
        self.main_areas = ["V1", "LM", "LI"]
        self.main_training_steps = ["naive", 45, 35, 30, 25, 20]
        self.get_ypred_indexes()
        self.get_all_tuning_curves()

    def get_ypred_indexes(self):
        all_ypred_index = []
        x_val = []
        for i, col in enumerate(self.df.columns):
            if "ypred" in col:
                all_ypred_index.append(i)
                x_val.append(float(col.split("_")[-1]))
        x_val = np.array(x_val)
        sorted_index = np.argsort(x_val)
        self.x_val = x_val[sorted_index]
        self.ypred_index = np.array(all_ypred_index)[sorted_index]

    def get_all_tuning_curves(self):
        tuning_curves = {}
        for main_area in self.main_areas:
            tc_list = []
            for main_training_step in self.main_training_steps:
                df = self.general_filters(self.df)
                df = df[(df.training_lvl == main_training_step) & (df.area_ID_str == main_area)]
                df = df.iloc[:, self.ypred_index]
                tc_list.append(df.values)
            tuning_curves[main_area] = np.array(tc_list, dtype = object)
        self.tuning_curves = tuning_curves

    def plot_slope_diff(self, ax, area, colormap, normalize_tuning=True, TO=90.0):
        activity = self.tuning_curves[area]
        colors = colormap(np.linspace(0, 1, len(self.main_training_steps) + 1))
        for t in range(len(self.main_training_steps)):
            act_t = activity[t]
            angles = np.linspace(0, 180, len(act_t[-1]))
            PO_index = np.argmax(act_t, axis=1)
            PO = angles[PO_index]
            if normalize_tuning:
                act_t = act_t/np.mean(act_t)
            slopes = np.diff(np.concatenate([act_t, act_t[:, 1][:, np.newaxis]], axis=1), axis=1)
            x_axis = PO - TO
            TO_index = np.argmin(np.abs(angles - TO))
            y_axis = slopes[:, TO_index]
            sort_index = np.argsort(x_axis)
            x_axis = x_axis[sort_index]
            y_axis = y_axis[sort_index]
            if t == 0:
                base_y = y_axis
                base_x = x_axis
            inter_y_base, _ = lowess(x_axis, np.interp(x_axis, base_x, base_y))
            smooth_y, _ = lowess(x_axis, y_axis)
            ax.plot(x_axis, smooth_y-inter_y_base, color=colors[t])
        return ax


    @staticmethod
    def general_filters(df, by_RF_max=False):
        # df = df[df.OTC_S2N >= 1]
        # df = df[df.RF_overlap == True]
        # if by_RF_max:
        #     df = df[df.RF_max >= .5]
        # else:
        #     df = df[df.RF_S2N >= 2.0]
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
        orientations = self.get_orientation_traininglvl_area(make_plot=False)
        bandwidths = self.get_bandwidth_traininglvl_area(make_plot=False)
        orientations = orientations[("naive", "V1")]
        bandwidths = bandwidths[("naive", "V1")]
        available_cells = len(orientations)
        sorted_ori_index = np.argsort(orientations)
        sorted_ori = orientations[sorted_ori_index]
        sorted_bandwidths = bandwidths[sorted_ori_index]
        skip_index = np.arange(0, available_cells, available_cells//hidden_dim)
        random_shift = np.random.randint(0, available_cells)
        skip_index = (skip_index + random_shift) % available_cells
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
    data_path = "/home/rodrigo/SSD/Projects/tmp_data/20240816203149_orituneData.pkl"
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