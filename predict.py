from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from soundevent import data
from torch.utils.data import DataLoader

from batdetect2.data.labels import ClassMapper
from batdetect2.models.detectors import DetectorModel
from batdetect2.train.augmentations import (
    add_echo,
    select_random_subclip,
    warp_spectrogram,
)
from batdetect2.train.dataset import LabeledDataset, get_files
from batdetect2.train.preprocess import PreprocessingConfig
from class_mapper import Mapper

class BatDetect2Predict:
    def __init__(self):
        self.parent_dir = "F:/bat/trainingBd2"
        self.checkpoint_dir = self.parent_dir + "/preprocessed"
        self.checkpoint = self.checkpoint_dir + "/lightning_logs/version_9/checkpoints/epoch=15-step=1600.ckpt"
        self.pre_proc_cfg_path = self.checkpoint_dir + "/config.json" 
        
    def prepare_data(self):
        root_dir = Path(self.parent_dir)
        data_dir = root_dir / "wav"
        files = get_files(root_dir / "preprocessed")
        print("prepare training data set")
        self.train_dataset = LabeledDataset(files)
        print("total size:", len(self.train_dataset))

        self.detector = DetectorModel.load_from_checkpoint(self.checkpoint)
        self.detector.eval()
        self.detector.load_preprocessing_config(self.pre_proc_cfg_path)
        
    def predict(self):        
        eventIdx = 1
        clipIdx = 555
        clip_annotation = self.train_dataset.get_clip_annotation(clipIdx)
        predictions = self.detector.compute_clip_predictions(clip_annotation.clip)
        print(f"Num predicted soundevents: {len(predictions.sound_events)} in clip [{clipIdx}]")      
        if len(predictions.sound_events) > eventIdx:
            print(f"sound_event[{eventIdx}]:")
            print(predictions.sound_events[eventIdx])
        if len(predictions.sequences) > eventIdx:
            print("sequences_events:")
            print(predictions.sequences[eventIdx])
        if len(predictions.tags) > eventIdx:
            print("tags:")
            print(predictions.tags[eventIdx])
#        if len(predictions.features) > eventIdx:
        print("------------- features:")
        print(predictions.features)

if __name__ == '__main__':
    BdPredict = BatDetect2Predict()
    BdPredict.prepare_data()
    BdPredict.predict()
