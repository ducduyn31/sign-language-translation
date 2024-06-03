from torch.utils.data import Dataset
from xtcocotools.coco import COCO
import os

class CocoWholeBodyDataset(Dataset):
    def __init__(self, data_path: str, annotation_path: str):
        self.data_path = data_path
        self.coco = COCO(annotation_path)
        self.anns_ids = self.coco.getAnnIds()
        
    def __len__(self):
        return len(self.anns_ids)
    
    def __getitem__(self, idx):
        img_id = self.coco.getImgIds()[idx]
        img = self.coco.loadImgs(img_id)[0]
        
        return img