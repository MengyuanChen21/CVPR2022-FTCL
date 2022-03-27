from tqdm import tqdm
import numpy as np
import torch


def train(args, model, pair_dataloader, criterion, optimizer):
    model.train()
    print("-------------------------------------------------------------------------------")
    device = args.device

    # train_process
    train_num_correct = 0
    train_num_total = 0

    loss_stack = []
    acm_loss_stack = []
    act_inst_loss_stack = []
    act_cont_loss_stack = []
    act_back_loss_stack = []
    guide_loss_stack = []
    att_loss_stack = []
    feat_loss_stack = []
    lcs_loss_stack = []
    fsd_loss_stack = []

    for input_feature_1, input_feature_2, vid_label_1, vid_label_2 in tqdm(pair_dataloader):

        vid_label_1 = vid_label_1.to(device)
        vid_label_2 = vid_label_2.to(device)
        input_feature_1 = input_feature_1.to(device)
        input_feature_2 = input_feature_2.to(device)

        output_1, output_2 = model(input_feature_1, input_feature_2)

        act_inst_cls_1, act_cont_cls_1, act_back_cls_1, act_inst_feat_1, act_cont_feat_1, act_back_feat_1, \
        temp_att_1, act_inst_cas_1, act_cas_1, act_cont_cas_1, act_back_cas_1, \
        candi_for_dp_1, act_candi_for_nw_1, bak_candi_for_nw_1 = output_1

        act_inst_cls_2, act_cont_cls_2, act_back_cls_2, act_inst_feat_2, act_cont_feat_2, act_back_feat_2, \
        temp_att_2, act_inst_cas_2, act_cas_2, act_cont_cas_2, act_back_cas_2, \
        candi_for_dp_2, act_candi_for_nw_2, bak_candi_for_nw_2 = output_2

        loss, loss_dict = criterion(act_inst_cls_1, act_cont_cls_1, act_back_cls_1, vid_label_1, temp_att_1,
                                    act_inst_feat_1, act_cont_feat_1, act_back_feat_1, act_inst_cas_1,
                                    candi_for_dp_1, act_candi_for_nw_1, bak_candi_for_nw_1,
                                    args,
                                    act_inst_cls_2, act_cont_cls_2, act_back_cls_2, vid_label_2, temp_att_2,
                                    act_inst_feat_2, act_cont_feat_2, act_back_feat_2, act_inst_cas_2,
                                    candi_for_dp_2, act_candi_for_nw_2, bak_candi_for_nw_2,
                                    )

        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            fg_score_1 = act_inst_cls_1[:, :args.action_cls_num]
            fg_score_2 = act_inst_cls_2[:, :args.action_cls_num]
            label_np_1 = vid_label_1.cpu().numpy()
            label_np_2 = vid_label_2.cpu().numpy()
            score_np_1 = fg_score_1.cpu().numpy()
            score_np_2 = fg_score_2.cpu().numpy()

            pred_np_1 = np.zeros_like(score_np_1)
            pred_np_2 = np.zeros_like(score_np_2)
            pred_np_1[score_np_1 >= args.cls_threshold] = 1
            pred_np_2[score_np_2 >= args.cls_threshold] = 1
            pred_np_1[score_np_1 < args.cls_threshold] = 0
            pred_np_2[score_np_2 < args.cls_threshold] = 0
            correct_pred_1 = np.sum(label_np_1 == pred_np_1, axis=1)
            correct_pred_2 = np.sum(label_np_2 == pred_np_2, axis=1)

            train_num_correct += np.sum(((correct_pred_1 == args.action_cls_num) *
                                         (correct_pred_2 == args.action_cls_num)))
            train_num_total += correct_pred_1.shape[0]

            loss_stack.append(loss.cpu().item())
            act_inst_loss_stack.append(loss_dict["act_inst_loss"])
            act_cont_loss_stack.append(loss_dict["act_cont_loss"])
            act_back_loss_stack.append(loss_dict["act_back_loss"])
            guide_loss_stack.append(loss_dict["guide_loss"])
            feat_loss_stack.append(loss_dict["feat_loss"])
            att_loss_stack.append(loss_dict["sparse_loss"])
            acm_loss_stack.append(loss_dict["acm_loss"])
            lcs_loss_stack.append(loss_dict["lcs_loss"])
            fsd_loss_stack.append(loss_dict["fsd_loss"])

    train_acc = train_num_correct / train_num_total

    train_log_dict = {"train_act_inst_cls_loss": np.mean(act_inst_loss_stack),
                      "train_act_cont_cls_loss": np.mean(act_cont_loss_stack),
                      "train_act_back_cls_loss": np.mean(act_back_loss_stack),
                      "train_guide_loss": np.mean(guide_loss_stack),
                      "train_feat_loss": np.mean(feat_loss_stack),
                      "train_att_loss": np.mean(att_loss_stack),
                      "train_acm_loss": np.mean(acm_loss_stack),
                      "train_lcs_loss": np.mean(lcs_loss_stack),
                      "train_fsd_loss": np.mean(fsd_loss_stack),
                      "train_loss": np.mean(loss_stack),
                      "train_acc": train_acc}

    print("\n")
    print("train_act_inst_cls_loss:{:.3f}  train_act_cont_cls_loss:{:.3f}".format(np.mean(act_inst_loss_stack),
                                                                                  np.mean(act_cont_loss_stack)))
    print("train_act_back_cls_loss:{:.3f}  train_att_loss:{:.3f}".format(np.mean(act_back_loss_stack),
                                                                         np.mean(att_loss_stack)))
    print("train_feat_loss:        {:.3f}  train_loss:{:.3f}".format(np.mean(feat_loss_stack), np.mean(loss_stack)))
    print("train acc:{:.3f}".format(train_acc))

    return train_log_dict
