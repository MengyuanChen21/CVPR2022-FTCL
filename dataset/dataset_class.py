import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random


def uniform_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if sample_len >= input_len > 1:
        sample_idxs = np.arange(input_len)
    else:
        if input_len == 1:
            sample_len = 2
        sample_scale = input_len / sample_len
        sample_idxs = np.arange(sample_len) * sample_scale
        sample_idxs = np.floor(sample_idxs)

    return input_feature[sample_idxs.astype(np.int), :]


def random_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len < sample_len:
        sample_idxs = np.random.choice(input_len, sample_len, replace=True)
        sample_idxs = np.sort(sample_idxs)
    elif input_len > sample_len:
        sample_idxs = np.arange(sample_len) * input_len / sample_len
        for i in range(sample_len - 1):
            sample_idxs[i] = np.random.choice(range(np.int(sample_idxs[i]), np.int(sample_idxs[i + 1] + 1)))
        sample_idxs[-1] = np.random.choice(np.arange(sample_idxs[-2], input_len))
    else:
        sample_idxs = np.arange(input_len)

    return input_feature[sample_idxs.astype(np.int), :]


def consecutive_sample(input_feature, sample_len):
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len >= sample_len:
        sample_idx = np.random.choice((input_len - sample_len))
        return input_feature[sample_idx:(sample_idx + sample_len), :]

    elif input_len < sample_len:
        empty_features = np.zeros((sample_len - input_len, input_feature.shape[1]))
        return np.concatenate((input_feature, empty_features), axis=0)


class TestDataset(Dataset):

    def __init__(self, args, phase="train", sample="random"):

        self.phase = phase
        self.sample = sample
        self.data_dir = args.data_dir
        self.sample_segments_num = args.sample_segments_num

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]

        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]

        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}

        self.action_class_num = args.action_cls_num

        self.get_label()

    def get_label(self):

        self.label_dict = {}
        for item_name in self.data_list:

            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0

            self.label_dict[item_name] = item_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_duration = self.gt_dict[vid_name]["duration"]
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name + ".npy"))

        vid_len = con_vid_feature.shape[0]

        if self.sample == "random":
            con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
        else:
            con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)

        con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32))

        vid_label_t = torch.as_tensor(vid_label.astype(np.float32))

        if self.phase == "train":
            return con_vid_spd_feature, vid_label_t
        else:
            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration


def build_test_dataset(args, phase="train", sample="random"):
    return TestDataset(args, phase, sample)


def build_train_dataset(args, phase="train", sample="random"):
    return TrainDataset(args, phase, sample)


class TrainDataset(Dataset):

    def __init__(self, args, phase="train", sample="random"):

        self.phase = phase
        self.sample = sample
        self.data_dir = args.data_dir
        self.sample_segments_num = args.sample_segments_num

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]

        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]

        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}

        self.action_class_num = args.action_cls_num

        self.label_dict = {}
        self.get_label()

        self.pos_pair_list = []
        self.neg_pair_list = []
        self.pair_list = []
        self.get_pair()

        self.feature_list = []
        self.get_feature()

    def get_label(self):

        for vid_name in self.data_list:

            vid_anns_list = self.gt_dict[vid_name]["annotations"]
            vid_label = np.zeros(self.action_class_num)
            for ann in vid_anns_list:
                ann_label = ann["label"]
                vid_label[self.action_class_idx_dict[ann_label]] = 1.0

            self.label_dict[vid_name] = vid_label

    def get_group(self):

        for idx in range(len(self.data_list)):

            vid_name = self.data_list[idx]
            group_idx = np.argwhere(self.label_dict[vid_name] == 1)
            for class_idx in group_idx:
                self.group_list[class_idx[0]].append(idx)

    def get_pair(self):

        for i in range(len(self.data_list)):
            label_0 = np.array(self.label_dict[self.data_list[i]])
            for j in range(i + 1, len(self.data_list)):
                label_1 = np.array(self.label_dict[self.data_list[j]])
                if (label_0 == label_1).all():
                    self.pos_pair_list.append([i, j])
                elif np.sum(label_0 * label_1) == 0:
                    self.neg_pair_list.append([i, j])

        # print('len of neg_pair_list:', len(self.neg_pair_list))
        # 18772
        # print('len of pos_pair_list:', len(self.pos_pair_list))
        # 861

        self.neg_pair_list = random.sample(self.neg_pair_list, len(self.pos_pair_list))

        self.neg_pair_list.extend(self.pos_pair_list)
        self.pair_list = self.neg_pair_list

    def get_feature(self):

        for vid_name in self.data_list:
            con_vid_feature = np.load(os.path.join(self.feature_dir, f'{vid_name}.npy'))

            if self.sample == "random":
                con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)

            self.feature_list.append(con_vid_spd_feature)

    def __len__(self):
        if self.phase == 'train':
            return len(self.pair_list)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):

        if self.phase == 'train':
            idx_1, idx_2 = self.pair_list[idx]

            feature_1 = torch.as_tensor(self.feature_list[idx_1].astype(np.float32))
            feature_2 = torch.as_tensor(self.feature_list[idx_2].astype(np.float32))

            label_1 = torch.as_tensor(self.label_dict[self.data_list[idx_1]].astype(np.float32))
            label_2 = torch.as_tensor(self.label_dict[self.data_list[idx_2]].astype(np.float32))

            return feature_1, feature_2, label_1, label_2

        else:
            vid_name = self.data_list[idx]
            vid_label = self.label_dict[vid_name]
            vid_duration = self.gt_dict[vid_name]["duration"]
            con_vid_feature = self.feature_list[idx]
            vid_len = con_vid_feature.shape[0]

            if self.sample == "random":
                con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)

            con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32))

            vid_label_t = torch.as_tensor(vid_label.astype(np.float32))

            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration


