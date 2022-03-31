import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm


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


def weights_init(model):
    if isinstance(model, nn.Conv2d):
        model.weights.data.normal_(0.0, 0.001)
    elif isinstance(model, nn.Linear):
        model.weights.data.normal_(0.0, 0.001)


def generate_pairs(vid_label):
    pos_pair = []
    neg_pair = []
    batch_size = vid_label.shape[0]

    for i in range(batch_size - 1):
        for j in range(i + 1, batch_size):
            if torch.sum(vid_label[i] * vid_label[j]) == 1 and vid_label[i].equal(vid_label[j]):
                pos_pair.append([i, j])
            elif torch.sum(vid_label[i] * vid_label[j]) == 0:
                neg_pair.append([i, j])

    sample_neg_pair = random.sample(neg_pair, len(pos_pair))
    return pos_pair, sample_neg_pair


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


class ACMLoss(nn.Module):

    def __init__(self, args, dataset="THUMOS14"):
        super(ACMLoss, self).__init__()

        self.dataset = dataset
        self.lamb1 = args.loss_lamb_1  # att_norm_loss param
        self.lamb2 = args.loss_lamb_2
        self.lamb3 = args.loss_lamb_3
        self.lamb_lcs = args.loss_lamb_lcs
        self.lamb_fsd = args.loss_lamb_fsd
        self.feat_margin = 50  # 50
        self.thres = args.thres
        self.lcs_len = args.lcs_len
        self.fsd_len = args.fsd_len

    def cls_criterion(self, inputs, label):
        # return - torch.mean(torch.sum(torch.log(inputs) * label, dim=1))
        return - torch.mean(torch.sum(torch.log(inputs + 1e-45) * label, dim=1))

    def forward(self, act_inst_cls, act_cont_cls, act_back_cls, vid_label, temp_att,
                act_inst_feat, act_cont_feat, act_back_feat, temp_cas,
                lcs_candi, fsd_act_candi, fsd_bak_candi, args):

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
        guide_loss = torch.sum(torch.abs(1 - temp_cas[:, :, -1] - temp_att[:, :, 0].detach()), dim=1).mean()

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
        sparse_loss = torch.sum(temp_att[:, :, :2], dim=1).mean()

        if self.dataset == "THUMOS14":
            cls_loss = 1.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            cls_loss = 5.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss

        add_loss = self.lamb1 * guide_loss + self.lamb2 * feat_loss + self.lamb3 * sparse_loss

        acm_loss = cls_loss + add_loss
        loss = acm_loss

        loss_dict = {"act_inst_loss": act_inst_loss.cpu().item(),
                     "act_cont_loss": act_cont_loss.cpu().item(),
                     "act_back_loss": act_back_loss.cpu().item(),
                     "guide_loss": guide_loss.cpu().item(),
                     "feat_loss": feat_loss.cpu().item(),
                     "sparse_loss": sparse_loss.cpu().item(),
                     "acm_loss": acm_loss.cpu().item(),
                     }

        if not args.test:
            pos_pair, neg_pair = generate_pairs(vid_label)

            if args.without_lcs or len(pos_pair) == 0:
                lcs_loss = torch.tensor(0)
            else:
                lcs_loss = self.get_lcs_loss(pos_pair, neg_pair, lcs_candi)

            if args.without_fsd or len(pos_pair) == 0:
                fsd_loss = torch.tensor(0)
            else:
                fsd_loss = self.get_fsd_loss(pos_pair, neg_pair, fsd_act_candi, fsd_bak_candi)

            ftcl_loss = self.lamb_lcs * lcs_loss + self.lamb_fsd * fsd_loss
            loss += ftcl_loss

            loss_dict.update({"lcs_loss": (self.lamb_lcs * lcs_loss).cpu().item(),
                              "fsd_loss": (self.lamb_fsd * fsd_loss).cpu().item()})

        return loss, loss_dict

    def get_fsd_loss(self, pos_pair, neg_pair, fsd_act_candi, fsd_bak_candi):
        act_bak_fsd = self.dp_for_fsd(neg_pair, fsd_act_candi, fsd_bak_candi, act_act=False)
        act_act_fsd = self.dp_for_fsd(pos_pair, fsd_act_candi, fsd_bak_candi, act_act=True)
        res = act_bak_fsd - act_act_fsd
        return res

    def dp_for_fsd(self, pair_list, act_candi, bak_candi, act_act=True):
        n_pair = len(pair_list)
        length = self.fsd_len
        fsd = torch.zeros(n_pair, device="cuda")
        for pair_idx in range(n_pair):
            candi_0 = act_candi[pair_list[pair_idx][0]]
            if act_act:
                candi_1 = act_candi[pair_list[pair_idx][1]]
            else:
                candi_1 = bak_candi[pair_list[pair_idx][1]]

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
            fsd[pair_idx] = C[-1, -1]
        return fsd.mean()

    def get_lcs_loss(self, pos_pair, neg_pair, lcs_candi):
        pos_lcs = self.dp_for_lcs(pos_pair, lcs_candi)
        neg_lcs = self.dp_for_lcs(neg_pair, lcs_candi)
        res = neg_lcs - pos_lcs
        return res

    def dp_for_lcs(self, pair_list, lcs_candi):
        n_pair = len(pair_list)
        lcs = torch.zeros(n_pair, device="cuda")
        length = self.lcs_len
        for pair_idx in tqdm(range(n_pair)):
            candi_0 = lcs_candi[pair_list[pair_idx][0]]
            candi_1 = lcs_candi[pair_list[pair_idx][1]]
            similar_matrix = get_similarity(candi_0, candi_1)
            scale_similar_matrix = torch.maximum(torch.zeros_like(similar_matrix), similar_matrix - 0.8) * 5
            C = torch.zeros((length + 1, length + 1), device="cuda")
            for i in range(1, length + 1):
                for j in range(1, length + 1):
                    if scale_similar_matrix[i - 1, j - 1] > self.thres:
                        C[i, j] = C[i - 1, j - 1] + scale_similar_matrix[i - 1, j - 1]
                    else:
                        C[i, j] = max(C[i - 1, j], C[i, j - 1])
            lcs[pair_idx] = C[-1, -1]
        return lcs.mean()
