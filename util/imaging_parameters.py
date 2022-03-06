import numpy as np


class Param:
    figure_size = (3.0, 3.0)
    config_dir = "C:/Users/user/PycharmProjects/rddetr/training/config/"
    tx_antenna_position = np.array([[-0.5, 0, 1.050],
                                    [-0.5, 0, 0.800],
                                    [-0.5, 0, 0.550],
                                    [-0.5, 0, 0.300],
                                    [0.5, 0, 1.050],
                                    [0.5, 0, 0.800],
                                    [0.5, 0, 0.550],
                                    [0.5, 0, 0.300]
                                    ])
    rx_antenna_position = np.array([[-0.3, 0, 1.050],
                                    [-0.1, 0, 1.050],
                                    [0.1, 0, 1.050],
                                    [0.3, 0, 1.050],
                                    [-0.3, 0, 0.300],
                                    [-0.1, 0, 0.300],
                                    [0.1, 0, 0.300],
                                    [0.3, 0, 0.300]
                                    ])
    num_tx = len(tx_antenna_position)
    num_rx = len(rx_antenna_position)
    volume_min_point = np.array([-1.5, 0.5, -1])
    volume_max_point = np.array([1.5, 3.5, 2])
    volume_num_step = np.array([100, 100, 100])
    y_min = -3
    y_max = 3
    y_min_filtered = -0.5
    y_max_filtered = 0.5
    snr_for_wiener_dB = -13
    image_intensity = 7
    offset = 1
    cnt = 1
    num_sample = 1024