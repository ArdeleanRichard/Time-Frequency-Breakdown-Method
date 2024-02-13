import matplotlib.pyplot as plt

from algorithms.TFBM import TFBM
from common.image_proc import apply_mask
from common.structs import Spectrum2D

import numpy as np
from sklearn.decomposition import PCA


def plot_data_result_mask(method_name, data, labelsMatrix, center_coords):
    fig= plt.figure(figsize=(24, 6))
    fig.suptitle(method_name, fontsize=16)
    ax = fig.add_subplot(1, 3, 1)

    ax.set_title("Initial Data")
    im = ax.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Mask")
    im = ax.imshow(labelsMatrix, aspect='auto', cmap='jet', interpolation='none')
    ax.invert_yaxis()
    plt.colorbar(im)

    for center_coord in center_coords:
        plt.scatter(center_coord[1], center_coord[0], s=1, c='black', marker='o')

    masked = apply_mask(data, labelsMatrix)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Segmentation")
    im = ax.imshow(masked, aspect='auto', cmap='jet', interpolation='none')
    plt.colorbar(im)
    ax.invert_yaxis()


    plt.show()


def TFBM_and_plot(data):
    tfbm = TFBM(data.T, threshold="auto", merge=True, aspect_ratio=1, merge_factor=15)
    tfbm.fit(verbose=True, timer=True)

    center_coords = [(pi.center_coords[1], pi.center_coords[0]) for pi in tfbm.packet_infos]

    plot_data_result_mask("TFBM", data, tfbm.merged_labels_data.T, center_coords)


def load_atoms_synthetic_data():
    data_folder = "./data/toy/"
    file = "atoms-2.csv"

    f = open(data_folder+file, "r")
    intro = f.readlines()[:5]
    f.close()

    timeValues = []
    for str_time in intro[1].split(","):
        timeValues.append(float(str_time))

    frequencyValues = []
    for str_time in intro[3].split(","):
        frequencyValues.append(float(str_time))

    data = np.loadtxt(data_folder + file, delimiter=",", dtype=float, skiprows=5)

    spectrumData = Spectrum2D(timeValues=np.array(timeValues), frequencyValues=frequencyValues, powerValues=data)

    return data, spectrumData


if __name__ == '__main__':
    data, spectrumData = load_atoms_synthetic_data()
    TFBM_and_plot(data)
