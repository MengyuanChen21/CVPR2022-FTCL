import torch
import torch.nn as nn
import numpy as np


def get_similarity(candi_0, candi_1):
    candi_0 = candi_0 / torch.linalg.vector_norm(candi_0, ord=2, dim=1, keepdim=True)
    candi_1 = candi_1 / torch.linalg.vector_norm(candi_1, ord=2, dim=1, keepdim=True)
    similarity = torch.mm(candi_0, candi_1.T)
    return similarity


def assign2Tensor(tensor, i, j, new_val):
    """ function to deal with tf.Tensors being non-assignable """
    # create mask
    mask = np.ones(tensor.shape, dtype=np.float32)
    # hack to assign a new value to tensor at position (i,j)
    mask[i, j] = 0
    mask = torch.tensor(mask, dtype=torch.float32, device=tensor.device)
    tensor = (tensor * mask) + (new_val * (1 - mask))
    return tensor


def contrastive_loss(x, eta=0.2):
    return torch.log(1 + torch.exp(x / eta))


def minGamma(inputs, gamma=0.1):
    """ continuous relaxation of min defined in the D3TW paper"""
    if gamma == 0:
        minG = torch.min(inputs)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi = torch.max(zi)
        log_sum_G = max_zi + torch.log(torch.sum(torch.exp(zi - max_zi)))  # + 1e-10)
        minG = -gamma * log_sum_G
    return minG


def maxGamma(inputs, gamma=0.1):
    """ continuous relaxation of max """
    if gamma == 0:
        maxG = torch.max(inputs)
    else:
        # log-sum-exp stabilization trick
        zi = (inputs / gamma)
        max_zi = torch.max(zi)
        log_sum_G = max_zi + torch.log(torch.sum(torch.exp(zi - max_zi)))  # + 1e-10)
        maxG = gamma * log_sum_G
    return maxG


class FTCLLoss(nn.Module):

    def __init__(self, args, dataset="THUMOS14"):
        super(FTCLLoss, self).__init__()

        self.dataset = dataset
        self.lamb1 = args.loss_lamb_1  # att_norm_loss param
        self.lamb2 = args.loss_lamb_2
        self.lamb3 = args.loss_lamb_3
        self.action_cls_num = args.action_cls_num
        self.batch_size = args.batch_size  # 16
        self.feature_dim = args.feature_dim

        self.lamb_lcs = args.loss_lamb_lcs
        self.lamb_fsd = args.loss_lamb_fsd
        self.feat_margin = 50  # 50
        self.thres = args.thres
        self.lcs_len = args.lcs_len
        self.fsd_len = args.fsd_len

    def cls_criterion(self, inputs, label):
        return - torch.mean(torch.sum(torch.log(inputs + 1e-10) * label, dim=1))

    def forward_once(self, act_inst_cls, act_cont_cls, act_back_cls, vid_label, temp_att=None,
                     act_inst_feat=None, act_cont_feat=None, act_back_feat=None, act_inst_cas=None):

        device = act_inst_cls.device
        batch_size = act_inst_cls.shape[0]

        act_inst_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        act_cont_label = torch.hstack((vid_label, torch.ones((batch_size, 1), device=device)))
        act_back_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))

        act_inst_label = act_inst_label / torch.sum(act_inst_label, dim=1, keepdim=True)
        act_cont_label = act_cont_label / torch.sum(act_cont_label, dim=1, keepdim=True)
        act_back_label = act_back_label / torch.sum(act_back_label, dim=1, keepdim=True)

        act_inst_loss = self.cls_criterion(act_inst_cls, act_inst_label)
        act_cont_loss = self.cls_criterion(act_cont_cls, act_cont_label)
        act_back_loss = self.cls_criterion(act_back_cls, act_back_label)

        # Guide Loss
        guide_loss = torch.sum(torch.abs(1 - act_inst_cas[:, :, -1] - temp_att[:, :, 0].detach()), dim=1).mean()

        # Feat Loss
        act_inst_feat_norm = torch.norm(act_inst_feat, p=2, dim=1)
        act_cont_feat_norm = torch.norm(act_cont_feat, p=2, dim=1)
        act_back_feat_norm = torch.norm(act_back_feat, p=2, dim=1)

        feat_loss_1 = self.feat_margin - act_inst_feat_norm + act_cont_feat_norm
        feat_loss_1[feat_loss_1 < 0] = 0
        feat_loss_2 = self.feat_margin - act_cont_feat_norm + act_back_feat_norm
        feat_loss_2[feat_loss_2 < 0] = 0
        feat_loss_3 = act_back_feat_norm
        feat_loss = torch.mean((feat_loss_1 + feat_loss_2 + feat_loss_3) ** 2)

        # Sparse Att Loss
        # att_loss = torch.sum(temp_att[:, :, 0], dim=1).mean() + torch.sum(temp_att[:, :, 1], dim=1).mean()
        sparse_loss = torch.sum(temp_att[:, :, :2], dim=1).mean()

        if self.dataset == "THUMOS14":
            cls_loss = 1.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            cls_loss = 5.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss

        add_loss = self.lamb1 * guide_loss + self.lamb2 * feat_loss + self.lamb3 * sparse_loss

        loss = cls_loss + add_loss

        loss_dict = {"act_inst_loss": act_inst_loss.cpu().item(),
                     "act_cont_loss": act_cont_loss.cpu().item(),
                     "act_back_loss": act_back_loss.cpu().item(),
                     "guide_loss": guide_loss.cpu().item(),
                     "feat_loss": feat_loss.cpu().item(),
                     "sparse_loss": sparse_loss.cpu().item(),
                     "acm_loss": loss.cpu().item()}

        return loss, loss_dict

    def get_fsd_loss(self, pair_label, fsd_act_candi_0, fsd_act_candi_1, fsd_bak_candi_0, fsd_bak_candi_1):
        bs = fsd_act_candi_0.shape[0]
        fsd = torch.zeros(bs, device="cuda")
        for i in range(bs):
            if pair_label[i] == 1:
                fsd[i] = self.dp_for_fsd(fsd_act_candi_0[i], fsd_act_candi_1[i])
            else:
                fsd[i] = self.dp_for_fsd(fsd_act_candi_0[i], fsd_bak_candi_1[i])

        pos_fsd = fsd * pair_label
        neg_fsd = fsd * (1-pair_label)
        pos_num = torch.sum(pair_label)
        neg_num = bs - pos_num
        res = torch.sum(neg_fsd) / (neg_num + 1e-10) - torch.sum(pos_fsd) / (pos_num + 1e-10)
        return res

    def dp_for_fsd(self, candi_0, candi_1):
        length = self.fsd_len

        half_feature_dim = int(candi_1.shape[-1] / 2)
        m = get_similarity(candi_0[:, :half_feature_dim], candi_1[:, :half_feature_dim])
        g = get_similarity(candi_0[:, half_feature_dim:], candi_1[:, half_feature_dim:])

        C = torch.zeros((length + 1, length + 1), device="cuda")
        for i in range(1, length + 1):
            for j in range(1, length + 1):
                neighbors = torch.stack([C[i - 1, j - 1],
                                         g[i - 1, j - 1] + C[i - 1, j],
                                         g[i - 1, j - 1] + C[i, j - 1]])
                new_val = m[i - 1, j - 1] + maxGamma(neighbors)
                C = assign2Tensor(C, i, j, new_val)
        return C[-1, -1]

    def get_lcs_loss(self, pair_label, lcs_candi_0, lcs_candi_1):
        bs = lcs_candi_0.shape[0]
        lcs = self.dp_for_lcs(lcs_candi_0, lcs_candi_1)
        pos_lcs = lcs * pair_label
        neg_lcs = lcs * (1-pair_label)

        pos_num = torch.sum(pair_label)
        neg_num = bs - pos_num
        res = torch.sum(neg_lcs) / (neg_num + 1e-10) - torch.sum(pos_lcs) / (pos_num + 1e-10)
        return res

    def dp_for_lcs(self, lcs_candi_0, lcs_candi_1):
        bs = lcs_candi_1.shape[0]
        lcs = torch.zeros(bs, device="cuda")
        length = self.lcs_len
        for idx in range(bs):
            candi_0 = lcs_candi_0[idx]
            candi_1 = lcs_candi_1[idx]
            similar_matrix = get_similarity(candi_0, candi_1)
            C = torch.zeros((length + 1, length + 1), device="cuda")
            for i in range(1, length + 1):
                for j in range(1, length + 1):
                    if similar_matrix[i - 1, j - 1] > self.thres:
                        C[i, j] = C[i - 1, j - 1] + similar_matrix[i - 1, j - 1]
                    else:
                        C[i, j] = max(C[i - 1, j], C[i, j - 1])
            lcs[idx] = C[-1, -1]
        return lcs

    def forward(self, act_inst_cls_0, act_cont_cls_0, act_back_cls_0, vid_label_0, temp_att_0, act_inst_feat_0,
                act_cont_feat_0, act_back_feat_0, act_inst_cas_0, lcs_candi_0, fsd_act_candi_0, fsd_bak_candi_0,
                args,
                act_inst_cls_1=None, act_cont_cls_1=None, act_back_cls_1=None, vid_label_1=None, temp_att_1=None,
                act_inst_feat_1=None, act_cont_feat_1=None, act_back_feat_1=None, act_inst_cas_1=None,
                lcs_candi_1=None, fsd_act_candi_1=None, fsd_bak_candi_1=None):

        loss_dict = {}

        if act_inst_cls_1 is not None:
            loss_0, loss_dict_0 = self.forward_once(act_inst_cls_0, act_cont_cls_0, act_back_cls_0, vid_label_0,
                                                    temp_att_0, act_inst_feat_0, act_cont_feat_0, act_back_feat_0,
                                                    act_inst_cas_0)
            loss_1, loss_dict_1 = self.forward_once(act_inst_cls_1, act_cont_cls_1, act_back_cls_1, vid_label_1,
                                                    temp_att_1, act_inst_feat_1, act_cont_feat_1, act_back_feat_1,
                                                    act_inst_cas_1)
            acm_loss = (loss_0 + loss_1) / 2.0

            loss_dict['act_inst_loss'] = (loss_dict_0['act_inst_loss'] + loss_dict_1['act_inst_loss']) / 2.0
            loss_dict['act_cont_loss'] = (loss_dict_0['act_cont_loss'] + loss_dict_1['act_cont_loss']) / 2.0
            loss_dict['act_back_loss'] = (loss_dict_0['act_back_loss'] + loss_dict_1['act_back_loss']) / 2.0
            loss_dict['guide_loss'] = (loss_dict_0['guide_loss'] + loss_dict_1['guide_loss']) / 2.0
            loss_dict['feat_loss'] = (loss_dict_0['feat_loss'] + loss_dict_1['feat_loss']) / 2.0
            loss_dict['sparse_loss'] = (loss_dict_0['sparse_loss'] + loss_dict_1['sparse_loss']) / 2.0
            loss_dict['acm_loss'] = (loss_dict_0['acm_loss'] + loss_dict_1['acm_loss']) / 2.0

            pair_label = torch.sum(vid_label_0 * vid_label_1, dim=1)
            pair_label[pair_label > 0] = 1

            if args.without_lcs:
                lcs_loss = torch.tensor(0)
            else:
                lcs_loss = self.get_lcs_loss(pair_label, lcs_candi_0, lcs_candi_1)

            if args.without_fsd:
                fsd_loss = torch.tensor(0)
            else:
                fsd_loss = self.get_fsd_loss(pair_label, fsd_act_candi_0, fsd_act_candi_1,
                                             fsd_bak_candi_0, fsd_bak_candi_1)

            ftcl_loss = self.lamb_lcs * lcs_loss + self.lamb_fsd * fsd_loss
            loss = acm_loss + ftcl_loss

            loss_dict.update({"lcs_loss": (self.lamb_lcs * lcs_loss).cpu().item(),
                              "fsd_loss": (self.lamb_fsd * fsd_loss).cpu().item()})

            return loss, loss_dict

        else:
            return self.forward_once(act_inst_cls_0, act_cont_cls_0, act_back_cls_0, vid_label_0,
                                     temp_att_0, act_inst_feat_0, act_cont_feat_0, act_back_feat_0, act_inst_cas_0)
