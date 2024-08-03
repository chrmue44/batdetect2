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

from soundevent.evaluation.encoding import create_tag_encoder
from soundevent.evaluation.tasks.sound_event_detection import _evaluate_clips

import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class BatDetect2Predict:
    def __init__(self):
        self.parent_dir = "F:/bat/trainingBd2"
        self.checkpoint_dir = self.parent_dir + "/preprocessed"
        self.checkpoint = self.checkpoint_dir + "/lightning_logs/version_9/checkpoints/epoch=15-step=1600.ckpt"
        self.pre_proc_cfg_path = self.checkpoint_dir + "/config.json" 
        self.mapper = Mapper()
        
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
        
    def predict_single(self):        
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
    
    def compute_cofusion_matrix(self, y_true, y_scores, tag_encoder, len_tags):
        # Convert y_true list to a NumPy array of floats for further processing
        y_true = np.array(y_true, dtype=np.float32)

        # Replace missing values (None) in y_true with a numerical class representing
        # the background
        background_class = len_tags  # Assign a unique class index to the background
        y_true_1 = np.nan_to_num(y_true, nan=len_tags)

        # Add a predicted score column for the background class
        # This is calculated as the complement of the maximum probability assigned to
        # any other class
        y_scores_1 = np.c_[y_scores, 1 - y_scores.max(axis=1)]

        # Get predicted classes by selecting the index with the highest probability for
        # each match
        y_pred = np.argmax(y_scores_1, axis=1)
        y_pred = np.array(y_pred, dtype=np.float32)

        # Generate labels for the confusion matrix, including the background class
        labels = [f"{tag.key}:{tag.value}" for tag in tag_encoder._tags] + [
        "background"
        ]
        
        #extract only the labels that are present in y_true (otherwise error in matplotlib)
        unique = np.unique(y_true_1)
        disp_labels = []
        for i in unique:
            disp_labels.append(labels[int(i)])

        fig, ax = plt.subplots(figsize=(5, 5))
        disp = ConfusionMatrixDisplay.from_predictions(y_true_1, y_pred, ax=ax, xticks_rotation="vertical", display_labels = disp_labels)
        fig.tight_layout()                
        return disp
    
    def predict_multiple(self):
        clip_annotations = [self.train_dataset.get_clip_annotation(i) for i in range(333)]
        clip_predictions = [
            self.detector.compute_clip_predictions(clip_annotation.clip)
            for clip_annotation in clip_annotations
        ]
        tags = self.mapper.create_tags()
        tag_encoder = create_tag_encoder(tags)
        evaluated_clips, y_true, y_scores = _evaluate_clips(
            clip_predictions, clip_annotations, tag_encoder
        )
        disp = self.compute_cofusion_matrix(y_true, y_scores, tag_encoder, len(tags))
     #   disp.plot()
        plt.show()
        
if __name__ == '__main__':
    BdPredict = BatDetect2Predict()
    BdPredict.prepare_data()
    BdPredict.predict_multiple()
