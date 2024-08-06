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
import numpy

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
 #   random.seed(worker_seed)

class BatDetect2Trainer:
    def __init__(self):
        self.parent_dir = "F:/bat/trainingBd2"
        self.test_set_portion = 0.1
        self.checkpoint_dir = self.parent_dir + "/preprocessed"
        self.epochs = 60
        self.learning_rate = 0.0003
#        self.checkpoint = self.checkpoint_dir + "/lightning_logs/version_3/checkpoints/epoch=19-step=2000.ckpt"
        self.pre_proc_cfg_path = self.checkpoint_dir + "/config.json" 
        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None
        self.train_dataloader = None

        
    def prepare_data(self):
        """read the prepared test data and split it to train, valid and test set"""
        print("start preparing trian data...")
        root_dir = Path(self.parent_dir)
        data_dir = root_dir / "wav"
        files = get_files(root_dir / "preprocessed")
    
        print("prepare training data set")
        self.train_dataset = LabeledDataset(files)
        print("total size:", len(self.train_dataset))

        train_set_size = int(len(self.train_dataset) * (1 - self.test_set_portion))   
        test_set_size = len(self.train_dataset) - train_set_size
        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.train_dataset, [train_set_size, test_set_size], generator=seed)

        train_set_size = int(len(self.train_dataset) * 0.8)   
        valid_set_size = len(self.train_dataset) - train_set_size
        seed = torch.Generator().manual_seed(142)
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset, [train_set_size, valid_set_size], generator=seed)
        print("sizes:   train set:{} validation set:{}  test set:{}".format(train_set_size, valid_set_size, test_set_size))        
        print(self.valid_dataset)

 #       g = torch.Generator()
 #       g.manual_seed(0)


        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=64,
            num_workers=4,
            persistent_workers=True,
#            worker_init_fn = seed_worker,
#            generator = g,
        )
            
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
#            shuffle=False,
            batch_size=64,
            num_workers=4,
            persistent_workers=True,
#            worker_init_fn = seed_worker,
#            generator = g,
        )

        self.test_dataloader = DataLoader(
            self.test_dataset
            ,shuffle=False,
           batch_size=64,
           num_workers=4,
           persistent_workers=True,
#            worker_init_fn = seed_worker,
#            generator = g,
        )


        self.detector = DetectorModel(class_mapper=Mapper(), learning_rate = self.learning_rate)
#        self.detector = DetectorModel.load_from_checkpoint(self.checkpoint)
        self.detector.eval()
        self.detector.load_preprocessing_config(self.pre_proc_cfg_path)
        self.trainer = pl.Trainer(
            default_root_dir=self.checkpoint_dir,
            limit_train_batches=100,
            max_epochs=self.epochs,
            log_every_n_steps=1,
        )
        
    def train(self):
        print("start training...")
        self.trainer.fit(self.detector, self.train_dataloader, self.valid_dataloader)

#        clip_annotation = self.train_dataset.get_clip_annotation(0)
        self.trainer.test(self.detector, dataloaders=self.test_dataloader) 
#        predictions = detector.compute_clip_predictions(clip_annotation.clip)
#        print(f"Num predicted soundevents: {len(predictions.sound_events)}")        

if __name__ == '__main__':
    BdTrainer = BatDetect2Trainer()
    BdTrainer.prepare_data()
    BdTrainer.train()
    input("Press Enter to continue...")
