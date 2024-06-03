import json
import os
import cv2
import lightning as L

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.array import load_frame_nums_to_4darray, pad_array_np
import utils.videotransforms as vt


class WLASLDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, annotations: str, batch_size=6, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.annotations = annotations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = transforms.Compose(
            [
                vt.RandomCrop(224),
                vt.RandomHorizontalFlip(),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                vt.CenterCrop(224),
            ]
        )

    def setup(self, stage):
        if stage in ("fit", None):
            self.train_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="train",
                transforms=self.train_transforms,
            )
            self.val_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="val",
                transforms=self.val_transforms,
            )
        elif stage == "test":
            self.test_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="test",
                transforms=self.val_transforms,
            )
        elif stage == "predict":
            self.predict_dataset = WLASLDataset(
                annotations_path=self.annotations,
                data_dir=self.data_dir,
                subset="train",
                transforms=self.train_transforms,
                max_size=1000,
            )

    def collate_fn(self, batch):
        videos = []
        pads = []
        labels = []
        names = []
        for video, pad, label, name in batch:
            videos.append(video)
            pads.append(pad)
            labels.append(label)
            names.append(name)
        videos = torch.stack(videos)
        return videos, pads, labels, names

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


class WLASLDataset(Dataset):
    def __init__(
        self, annotations_path: str, data_dir: str, subset: str, max_size=None, transforms=None, cache_folder='.cache/'
    ):
        self.data_dir = data_dir
        self.annotations_path = annotations_path
        self.transforms = transforms
        self.n_classes = 2000
        self.frames_per_clip = 64
        self.max_size = max_size
        self.cache_folder = cache_folder

        assert subset in [
            "train",
            "val",
            "test",
        ], f"subset must be one of ['train', 'val', 'test']"

        self.subset = subset
        os.makedirs(self.cache_folder, exist_ok=True)
        self.data = self._prepare_and_cache(annotations_path, data_dir, subset, max_size)

    def _prepare_and_cache(self, annotations_path, data_dir, subset, max_size):
        cache_file = os.path.join(self.cache_folder, f"{subset}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)
        else:
            data = self._prepare_dataset(annotations_path, data_dir, subset, max_size)
            torch.save(data, cache_file)
            return data
        
    def _prepare_dataset(self, annotations_path: str, data_dir: str, subset: str, max_size):
        dataset = []

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        count = 0

        for k, v in annotations.items():
            if max_size and count >= max_size:
                break
            if v["subset"] != subset:
                continue

            video_path = os.path.join(data_dir, f"{k}.mp4")

            if not os.path.exists(video_path):
                continue
            n_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))

            action = v["action"][0]
            frame_start = v["action"][1]
            frame_end = v["action"][2]
            frame_count = frame_end - frame_start

            label = np.zeros((self.n_classes, n_frames), dtype=np.float32)
            for l in range(n_frames):
                label[action][l] = 1

            dataset.append(
                {
                    "name": k,
                    "video_path": video_path,
                    "action": label,
                    "start": (
                        frame_start
                        if len(k) == 6 and self.subset in ["train", "val"]
                        else 0
                    ),
                    "frames": (
                        frame_count if self.subset in ["train", "val"] else n_frames
                    ),
                }
            )
            count += 1

        return dataset

    def __len__(self):
        return len(self.data)

    
    def _load_video(self, video_path, vlen, num_frames, index_setting=['consecutive', 'pad', 'central', 'pad']):
        vlen = vlen - 2 # Bug
        
        selected_indices, pad = self._get_selected_indices(vlen, num_frames=num_frames, setting=index_setting)
        video_arrays = load_frame_nums_to_4darray(video_path, selected_indices)
        
        if pad is not None:
            video_arrays = pad_array_np(video_arrays, pad)
        return video_arrays, selected_indices, pad
            
    def _get_selected_indices(self, vlen, num_frames=64, setting=['consecutive', 'pad', 'central', 'pad']):
        pad = None
        assert len(setting) == 4
        
        train_p, train_m, test_p, test_m = setting

        selected_indices = np.arange(vlen)
        
        if num_frames > 0:
            if self.subset == 'train':
                if vlen > num_frames:
                    if train_p == 'consecutive':
                        start = np.random.randint(0, vlen - num_frames, 1)[0]
                        selected_indices = np.arange(start, start + num_frames)
                    elif train_p == 'random':
                        selected_indices = np.arange(vlen)
                        np.random.shuffle(selected_indices)
                        selected_indices = selected_indices[:num_frames]
                        selected_indices = np.sort(selected_indices)
                    else:
                        selected_indices = np.arange(0, vlen)
                elif vlen < num_frames:
                    if train_m == 'pad':
                        remaining = num_frames - vlen
                        selected_indices = np.arange(0, vlen)
                        pad_left = np.random.randint(0, remaining, 1)[0]
                        pad_right = remaining - pad_left
                        pad = (pad_left, pad_right)
                    else:
                        selected_indices = np.arange(0, vlen)
                else:
                    selected_indices = np.arange(0, vlen)

            else:
                if vlen >= num_frames:
                    start = 0
                    if test_p == 'central':
                        start = (vlen - num_frames) // 2
                    elif test_p == 'start':
                        start = 0
                    elif test_p == 'end':
                        start = vlen - num_frames
                    selected_indices = np.arange(start, start + num_frames)
                else:
                    remaining = num_frames - vlen
                    selected_indices = np.arange(0, vlen)
                    if test_m == 'pad':
                        pad_left = remaining // 2
                        pad_right = remaining - pad_left
                        pad = (pad_left, pad_right)
                    elif test_m == 'start_pad':
                        pad_left = 0
                        pad_right = remaining - pad_left
                        pad = (pad_left, pad_right)
                    elif test_m == 'end_pad':
                        pad_left = remaining - pad_right
                        pad_right = remaining - pad_left
                        pad = (pad_left, pad_right)
                    else:
                        selected_indices = np.arange(0, vlen)
        
        return selected_indices, pad
    
    # def _get_label(self, label, count):
        # if self.subset in ["test"]:
        #     return torch.from_numpy(label)
        # label = label[:, 0]
        # label = np.tile(label, (count, 1)).transpose((1, 0))
        # return torch.from_numpy(label)
        

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item["name"]
        video_path = item["video_path"]
        action = item["action"]
        frames_count = item["frames"]
        # start = item["start"]

        video, selected_indices, pad = self._load_video(video_path, frames_count, self.frames_per_clip)
        video = torch.tensor(video).float()
        # video /= 255.0
        # video = video.permute(0, 3, 1, 2) # T, C, H, W
        label = action.argmax(axis=0)[0]
        
        return video, pad, label, name
        

