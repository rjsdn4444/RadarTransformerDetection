import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import util
from util.imaging_parameters import Param as p
from util.radar_imaging import RadarImaging_filtering

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Imaging:
    def __init__(self):
        self.imaging = RadarImaging_filtering(tx_antenna_position=p.tx_antenna_position,
                                              rx_antenna_position=p.rx_antenna_position,
                                              config_dir=p.config_dir, num_tx_antenna=p.num_tx,
                                              num_rx_antenna=p.num_rx)
        self.imaging.set_volume(min_point=p.volume_min_point, max_point=p.volume_max_point,
                                num_step=p.volume_num_step)
        self.imaging.add_slice(type='xy', offset=1,
                               min_point=np.array([p.volume_min_point[0], p.volume_min_point[1]]),
                               max_point=np.array([p.volume_max_point[0], p.volume_max_point[1]]),
                               num_step=np.array([p.volume_num_step[0], p.volume_num_step[1]]))
        self.radar_pulse = self.imaging.set_entire_matched_filter("C:/Users/user/PycharmProjects/rddetr/pulses_long/")

    def radar_box_imaging(self, radar_signal, prediction_boxes, target_boxes, process_count, output_dir, predict):
        if process_count % 50 == 0:
            plt.ion()
            color = ['b', 'g']
            image_boxes = prediction_boxes
            num_slices = len(self.imaging._slices)
            num_object = len(image_boxes)
            fig, axes = plt.subplots(1, num_slices, figsize=(p.figure_size[0] * num_slices, p.figure_size[1]))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            images = []
            radar_image_list = self.imaging.get_radar_slice_image(radar_signal)
            for ax, (type, extent, _, _), radar_image in zip(axes, self.imaging._slices, radar_image_list):
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xlabel(type[0])
                ax.set_ylabel(type[1])
                images.append(ax.imshow(np.transpose(radar_image), origin='lower', extent=extent))
                if predict:
                    for objects in range(num_object):
                        self.box_imaging(images, ax, image_boxes[objects],
                                         image_labeling='prediction', color=color[objects],
                                         prediction=True)
                self.box_imaging(images, ax, target_boxes, image_labeling='target', color='r', prediction=False)
            plt.legend()
            fig.savefig(output_dir + '{}.png'.format(process_count))
            fig.show()

    def box_imaging(self, images, ax, image_boxes, image_labeling, color, prediction):
        if prediction:
            images.append(ax.plot([image_boxes[0], image_boxes[0]],
                                  [image_boxes[1], image_boxes[3]], c=color, label=image_labeling))
            images.append(ax.plot([image_boxes[0], image_boxes[2]],
                                  [image_boxes[1], image_boxes[1]], c=color))
            images.append(ax.plot([image_boxes[0], image_boxes[2]],
                                  [image_boxes[3], image_boxes[3]], c=color))
            images.append(ax.plot([image_boxes[2], image_boxes[2]],
                                  [image_boxes[1], image_boxes[3]], c=color))
        else:
            images.append(ax.plot([image_boxes[0][0], image_boxes[0][0]],
                                  [image_boxes[0][1], image_boxes[0][3]], c=color, label=image_labeling))
            images.append(ax.plot([image_boxes[0][0], image_boxes[0][2]],
                                  [image_boxes[0][1], image_boxes[0][1]], c=color))
            images.append(ax.plot([image_boxes[0][0], image_boxes[0][2]],
                                  [image_boxes[0][3], image_boxes[0][3]], c=color))
            images.append(ax.plot([image_boxes[0][2], image_boxes[0][2]],
                                  [image_boxes[0][1], image_boxes[0][3]], c=color))

    def wiener_filtering(self, radar_signal):
        stretched_signal = np.zeros((p.num_tx, p.num_rx, p.num_sample * 2))
        stretched_pulse = np.zeros((p.num_tx, p.num_rx, p.num_sample * 2))
        filtered_signal = np.zeros((p.num_tx, p.num_rx, p.num_sample))
        for tx in range(p.num_tx):
            for rx in range(p.num_rx):
                stretched_signal[tx][rx] = np.pad(radar_signal[tx][rx],
                                                  (0, len(radar_signal[tx][rx]) * 2 - len(radar_signal[tx][rx])),
                                                  'constant', constant_values=0)
                stretched_pulse[tx][rx] = np.pad(self.radar_pulse[str(tx) + str(rx)],
                                                 (0, len(radar_signal[tx][rx]) * 2 - len(self.radar_pulse[str(tx) + str(rx)])),
                                                 'constant', constant_values=0)
                filtered_signal[tx][rx] = self.imaging.convert_wiener_deconv(stretched_pulse[tx][rx],
                                                                             stretched_signal[tx][rx],
                                                                             snr_db=p.snr_for_wiener_dB)
        return filtered_signal

