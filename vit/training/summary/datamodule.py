# -*- coding: utf-8 -*-
import os
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


class SummaryDataset(Dataset):
    def __init__(self, video_names, directory, video_features_file, max_seq_len=250):
        self.directory = directory
        self.video_names = video_names
        self.video_features_file = video_features_file
        self.max_seq_len = max_seq_len

        # Because we can't use DDP with IterableDataset,
        # data must be pre-chunked to combat OOM.
        self.label_files = self.prefetch_label_files()
        self.data_size, self.index_to_chunk, self.labels = self.prefetch_and_index()

    def prefetch_label_files(self):

        name_set = set(self.video_names)

        label_files = defaultdict(list)

        for label_file in Path(self.directory).glob(f"**/*.json"):

            file_name = label_file.stem

            # 예시: [KBS]kim370_대법원 업무 과부하…상고 법원이 대안_18567498.json
            # annotator id 제거하면 비디오 이름 추출.
            # 파일 이름 reverse ([::-1]) 후 "_" 찾음.
            annotator_id_index = len(file_name) - file_name[::-1].find("_") - 1
            video_name = file_name[:annotator_id_index]

            if video_name in name_set:
                label_files[video_name].append(label_file)

        return label_files

    def prefetch_and_index(self):
        index = 0
        index_to_chunk = {}
        all_labels = {}

        for video_name in self.video_names:

            if video_name == "news_footage_1710":
                continue

            labels = self.extract_label(video_name)
            all_labels[video_name] = labels

            chunk_count = math.ceil(len(labels[0]) / self.max_seq_len)
            for chunk_index in range(0, chunk_count):
                index_to_chunk[index + chunk_index] = (video_name, chunk_index)

            index += chunk_count

        return index, index_to_chunk, all_labels

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):

        video_name, chunk_index = self.index_to_chunk[index]
        start = chunk_index * self.max_seq_len
        end = start + self.max_seq_len
        max_seq_len = end-start
        print(max_seq_len)
        with h5py.File(self.video_features_file, "r") as rf:

            labels = self.labels[video_name]
            video_features = rf[video_name][()][: len(labels[0])][start:end]
            labels = labels[:, start:end]
            return video_name, video_features, labels

    def extract_label(self, video_name):

        label_files = self.label_files[video_name]
        labels = []

        for label_file in label_files:

            with open(label_file, "r") as rf:
                data = json.load(rf)

            metadata = data["metadata"]
            video_length = math.ceil(metadata["length"])
            annotator_label = np.zeros((video_length,))

            for timeline in data["timelines"]:
                for time_index in range(timeline["start"], timeline["end"] + 1):
                    # annotator_label[time_index] += 1
                    if time_index < video_length:
                        annotator_label[time_index] = 1

            labels.append(annotator_label)

        labels = np.array(labels)
        return labels


def get_video_names(mp4_dir):
        # mp4 파일이 있는 디렉토리 경로
        # mp4_dir = "/workspace/EmotionShortForm/aihub/2.Validation/Video_data/VS_유튜브_04"

        video_names = []
        for filename in os.listdir(mp4_dir):
            if filename.endswith('.mp4'):
                name = os.path.splitext(filename)[0]
                video_names.append(name)
        return video_names


# +
mp4_dir = "/workspace/EmotionShortForm/data_AIHub/2.Validation/Video_data/VS_유튜브_04"
videos = get_video_names(mp4_dir)


data_directory = '/workspace/EmotionShortForm/data_AIHub/2.Validation/Labeling_data'
video_features_file = '/workspace/EmotionShortForm/data_AIHub/2.Validation/Video_data/VS_유튜브_04.h5'
sd = SummaryDataset(videos, data_directory, video_features_file)
sd[0]
# -

sd[2][2].shape

    dl = DataLoader(sd, batch_size=1)
    for d in dl:
        print(d)
        break

d[1].shape

d[2].shape

video_name, image_features, labels = d
video_name = video_name[0]
image_features = image_features.squeeze(0)
labels = labels.squeeze(0)

video_name

image_features.shape

labels.shape

import torch
score = torch.sum(labels, dim=0)
score = torch.min(
    score,
    torch.ones(score.shape[0],).to(score.device),
)

score.shape

# +
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch import optim
# from torchmetrics import F1
from transformers import ViTModel


class SummaryModel(LightningModule):
    def __init__(self, hidden_dim=768, individual_logs=None):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.scorer = nn.Linear(hidden_dim, 1)
        # self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        # self.train_f1 = F1()
        # self.val_f1 = F1()
        # self.test_f1 = F1()
        self.individual_logs = individual_logs
        self.tta_logs = defaultdict(list)

    def forward(self, x):
        x = self.vit(x).pooler_output
        x = self.scorer(x)
        # x = self.sigmoid(x)
        return x

    def run_batch(self, batch, batch_idx, metric, training=False):
        video_name, image_features, labels = batch
        video_name = video_name[0]
        image_features = image_features.squeeze(0)
        labels = labels.squeeze(0)

        # Score - aggregated labels.
        score = torch.sum(labels, dim=0)
        score = torch.min(
            score,
            torch.ones(
                score.shape[0],
            ).to(score.device),
        )
        out = self(image_features).squeeze(1)
        try:
            loss = self.loss(out.double(), score)
            preds = (torch.sigmoid(out) > 0.7).int()
            metric.update(preds, score.int())
            f1 = metric.compute()
            tp, fp, tn, fn = metric._get_final_stats()
            self.tta_logs[video_name].append((tp.item(), fp.item(), fn.item()))
        except Exception as e:
            print(e)
            loss = 0
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.train_f1, training=True)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_f1", self.train_f1.compute())
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.val_f1)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log("val_f1", self.val_f1.compute())
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss = self.run_batch(batch, batch_idx, self.test_f1)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        f1 = self.test_f1.compute()
        self.log("test_f1", f1)
        tp, fp, tn, fn = self.test_f1._get_final_stats()
        print(f"\nTest f1: {f1}, TP: {tp}, FP: {fp}, TN: {tn}, fn: {fn}")
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
# -


