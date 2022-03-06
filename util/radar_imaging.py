import numpy as np
import pyvista as pv
import math
import os
from pathlib import Path
import csv


class RadarImaging_filtering:
    def __init__(self, tx_antenna_position: np.ndarray, rx_antenna_position: np.ndarray, config_dir,
                 num_tx_antenna: int = 1, num_rx_antenna: int = 1):
        self._tx_antenna_position = tx_antenna_position
        self._rx_antenna_position = rx_antenna_position
        self._num_tx_antenna = num_tx_antenna
        self._num_rx_antenna = num_rx_antenna
        self._config_dir = config_dir
        self._sample_time = np.load(self._config_dir + '%08d.npz' % 1)['z'][0]
        self._start_sample_time = self._sample_time[0]
        self._sample_spacing = self._sample_time[1] - self._start_sample_time
        self._num_sample = self._sample_time.size

        self._speed_of_light = 0.299792458
        self._volume_sample_time_delay = None
        self._volume_sample_time_delay_invalid = None
        self._slices = []
        self._pyvista_uniform_grid = None

    def set_volume(self, min_point: np.ndarray, max_point: np.ndarray, num_step: np.ndarray):
        self._volume_sample_time_delay, self._volume_sample_time_delay_invalid, step_size = \
            self._set_area(min_point, max_point, num_step)
        self._pyvista_uniform_grid = pv.UniformGrid()
        self._pyvista_uniform_grid.dimensions = num_step
        self._pyvista_uniform_grid.spacing = step_size
        self._pyvista_uniform_grid.origin = min_point
        self._pyvista_uniform_grid.point_arrays['values'] = np.zeros(np.ndarray.prod(num_step))

    def add_slice(self, type: str, offset: float, min_point: np.ndarray, max_point: np.ndarray, num_step: np.ndarray):
        extent = (min_point[0], max_point[0], min_point[1], max_point[1])
        if type == 'xy':
            min_point = np.array([min_point[0], min_point[1], offset])
            max_point = np.array([max_point[0], max_point[1], offset])
            num_step = np.array([num_step[0], num_step[1], 1])
        elif type == 'yz':
            min_point = np.array([offset, min_point[0], min_point[1]])
            max_point = np.array([offset, max_point[0], max_point[1]])
            num_step = np.array([1, num_step[0], num_step[1]])
        elif type == 'zx':
            min_point = np.array([min_point[1], offset, min_point[0]])
            max_point = np.array([max_point[1], offset, max_point[0]])
            num_step = np.array([num_step[1], 1, num_step[0]])

        sample_time_delay, sample_time_delay_invalid, _ = self._set_area(min_point, max_point, num_step)
        self._slices.append((type, extent, sample_time_delay, sample_time_delay_invalid))

    def _set_area(self, min_point: np.ndarray, max_point: np.ndarray, num_step: np.ndarray):
        x_space, x_step_size = np.linspace(min_point[0], max_point[0], num_step[0], retstep=True)
        y_space, y_step_size = np.linspace(min_point[1], max_point[1], num_step[1], retstep=True)
        z_space, z_step_size = np.linspace(min_point[2], max_point[2], num_step[2], retstep=True)
        step_size = np.array([x_step_size, y_step_size, z_step_size])
        x_grid, y_grid, z_grid = np.meshgrid(x_space, y_space, z_space, indexing='ij')
        area_grid = np.stack((x_grid, y_grid, z_grid), axis=3)
        tx_distance = np.linalg.norm(
            area_grid - self._tx_antenna_position[:self._num_tx_antenna, np.newaxis, np.newaxis, np.newaxis, :],
            axis=4)
        rx_distance = np.linalg.norm(
            area_grid - self._rx_antenna_position[:self._num_rx_antenna, np.newaxis, np.newaxis, np.newaxis, :],
            axis=4)
        distance = tx_distance[:, np.newaxis, ...] + rx_distance

        sample_time_delay = np.round(((distance / self._speed_of_light) - self._start_sample_time) /
                                     self._sample_spacing).astype(np.int)
        sample_time_delay_invalid = (sample_time_delay < 0) | (sample_time_delay >= self._num_sample)
        sample_time_delay[sample_time_delay_invalid] = 0

        return sample_time_delay, sample_time_delay_invalid, step_size

    def get_radar_slice_image(self, radar_signal):
        radar_image_list = []
        for type, _, sample_time_delay, sample_time_delay_invalid in self._slices:
            radar_image = self._get_radar_image(sample_time_delay, sample_time_delay_invalid, radar_signal)
            if type == 'xy':
                radar_image = radar_image[:, :, 0]
            elif type == 'yz':
                radar_image = radar_image[0, :, :]
            elif type == 'zx':
                radar_image = np.transpose(radar_image[:, 0, :])
            radar_image_list.append(radar_image)

        return radar_image_list

    def _get_radar_image(self, sample_time_delay: np.ndarray, sample_time_delay_invalid: np.ndarray, radar_signal):
        # radar_signal = self.get_radar_signal(cnt)
        numpy_radar_signal = radar_signal.squeeze().cpu().detach().numpy()
        tx_antenna_index = np.arange(self._num_tx_antenna)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        rx_antenna_index = np.arange(self._num_rx_antenna)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        tx_rx_radar_image = np.array(numpy_radar_signal[tx_antenna_index, rx_antenna_index, sample_time_delay])
        tx_rx_radar_image[sample_time_delay_invalid] = 0
        radar_image = np.sum(tx_rx_radar_image, axis=(0, 1))
        radar_image = np.abs(radar_image)

        return radar_image

    def convert_wiener_deconv(self, stretched_radar_pulse, radar_signal, snr_db):
        radar_pulse_time = stretched_radar_pulse
        radar_signal_time = radar_signal
        radar_pulse_freq = np.fft.fft(radar_pulse_time)
        radar_signal_freq = np.fft.fft(radar_signal_time)
        radar_pulse_freq[0] = 0
        snr = math.pow(10, snr_db / 10)
        wiener_deconv_filter_freq = np.conj(radar_pulse_freq) / (np.square(np.absolute(radar_pulse_freq)) + 1 / snr)
        output_freq = radar_signal_freq * wiener_deconv_filter_freq
        output = np.fft.ifft(output_freq)
        output = output[: output.size // 2]

        return output

    def set_entire_matched_filter(self, pulse_dir: str):
        file_list = os.listdir(pulse_dir)
        port_radar_pulse = {}
        sps = 35859161088
        for file_name in file_list:
            port_tx = int(file_name.split('_')[0]) - 1
            port_rx = int(file_name.split('_')[1]) - 1
            port_combination = str(port_tx) + str(port_rx)
            with open(str(Path(pulse_dir + file_name).resolve()), 'r', newline='') as f:
                reader = csv.reader(f)
                radar_pulse = np.array([list(map(float, row)) for row in reader])
                file_sps = int(file_name.split('_')[2].split('.')[0])
                file_spacing = (10 ** 9) / file_sps
            if radar_pulse[0].size > 1:
                radar_pulse_x = radar_pulse[:, 0]
                radar_pulse_y = radar_pulse[:, 1]
            else:
                radar_pulse_x = np.arange(radar_pulse.size) * file_spacing
                radar_pulse_y = radar_pulse[:, 0]
            sample_spacing = (10 ** 9) / sps
            x = np.arange(radar_pulse_x[0], radar_pulse_x[-1], sample_spacing)
            interp_radar_pulse = np.interp(x, radar_pulse_x, radar_pulse_y)
            centered_radar_pulse = np.subtract(interp_radar_pulse, np.mean(interp_radar_pulse))
            normalized_radar_pulse = np.divide(centered_radar_pulse, np.linalg.norm(centered_radar_pulse))
            port_radar_pulse[port_combination] = normalized_radar_pulse[:self._num_sample]

        return port_radar_pulse
