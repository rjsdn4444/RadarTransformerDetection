import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from util.imaging import Imaging


class RadarDataset(Dataset):
    def __init__(self, radar_dir, ann_file):
        self.target_processing = TargetProcessing(dataset_imaging=False, radar_imaging_size=(3, 3, 2))
        self.annotation_files = COCO(ann_file)
        self.ids = list(sorted(self.annotation_files.imgs.keys()))
        self.radar_dir = radar_dir
        self.ann_file = ann_file

    def _load_radar_signal(self, id: int):
        path = self.annotation_files.loadImgs(id)[0]["file_name"]
        return torch.from_numpy(np.load(self.radar_dir + '/' + path))

    def _load_target(self, id: int):
        return self.annotation_files.loadAnns(self.annotation_files.getAnnIds(id))

    def __getitem__(self, idx):
        index = self.ids[idx]
        radar_signal = self._load_radar_signal(index)
        target = self._load_target(index)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        radar_signal, target = self.target_processing(radar_signal=radar_signal, target=target)
        return radar_signal, target

    def __len__(self):
        return len(self.ids)


class TargetProcessing:
    def __init__(self, dataset_imaging, radar_imaging_size):
        self.h, self.w, self.z = radar_imaging_size
        self.dataset_imaging = dataset_imaging
        if self.dataset_imaging:
            self.imaging = Imaging()

    def __call__(self, radar_signal, target):
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        annotations = target["annotations"]
        annotations = [objects for objects in annotations if 'iscrowd' not in objects or objects["iscrowd"] == 0]
        boxes = [objects["3d_bbox"] for objects in annotations]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 6)
        boxes = boxes + torch.tensor([[0, 0, 0, 0.8, 0.8, 0.8]], dtype=torch.float32)
        boxes = (boxes - torch.tensor([[-1.5, 0.5, 0, 0, 0, 0]], dtype=torch.float32)) / \
                torch.tensor([self.w, self.h, self.z] * 2, dtype=torch.float32)
        keypoints = [objects["keypoint"] for objects in annotations]
        num_points = len(keypoints[0])
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32).reshape(-1, num_points * 3)
        keypoints = (keypoints - torch.tensor([[-1.5, 0.5, 0] * num_points], dtype=torch.float32)) / \
                    torch.tensor([self.w, self.h, self.z] * num_points, dtype=torch.float32)
        classes = [objects["category_id"] for objects in annotations]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = True
        target = {}
        target["boxes"] = boxes
        target["keypoints"] = keypoints
        target["labels"] = classes
        target["image_id"] = image_id
        iscrowd = torch.tensor([objects["iscrowd"] if "iscrowd" in objects else 0 for objects in annotations])
        target["iscrowd"] = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(self.h), int(self.w), int(self.z)])

        return radar_signal, target


def build_dataset(dataset, base_dir):
    PATHS = {
        "train": (base_dir + "training/raw", base_dir + "annotation_keypoint_train.json"),
        "val": (base_dir + "testing/raw", base_dir + "annotation_keypoint_test.json"),
        "test": (base_dir + "testing/raw", base_dir + "annotation_keypoint_test.json")
    }
    radar_signal_folder, annotation_file = PATHS[dataset]
    data = RadarDataset(radar_signal_folder, annotation_file)
    return data
