import numpy as np
from tqdm import tqdm
from utils.net_evaluation import ANETDetection, upgrade_resolution, get_proposal_oic, nms, result2json
import os
import json


def test(args, model, dataloader, criterion):
    model.eval()
    print("-------------------------------------------------------------------------------")
    device = args.device
    save_dir = args.save_dir

    test_num_correct = 0
    test_num_total = 0

    acm_loss_stack = []
    act_inst_loss_stack = []
    act_cont_loss_stack = []
    act_back_loss_stack = []
    guide_loss_stack = []
    att_loss_stack = []
    feat_loss_stack = []

    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

    test_pred_score_stack = []
    test_vid_label_stack = []
    test_tmp_data_log_dict = {}

    for vid_name, input_feature, vid_label_t, vid_len, vid_duration in tqdm(dataloader):

        input_feature = input_feature.to(device)
        vid_label_t = vid_label_t.to(device)
        vid_len = vid_len[0].cpu().numpy()
        t_factor = (args.segment_frames_num * vid_len) / (
                args.frames_per_sec * args.test_upgrade_scale * input_feature.shape[1])

        act_inst_cls, act_cont_cls, act_back_cls, \
        act_inst_feat, act_cont_feat, act_back_feat, \
        temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas,\
        lcs_candi, fsd_act_candi, fsd_bak_candi = model(input_feature)

        loss, loss_dict = criterion(act_inst_cls, act_cont_cls, act_back_cls, vid_label_t, temp_att,
                                    act_inst_feat, act_cont_feat, act_back_feat, act_inst_cas,
                                    lcs_candi, fsd_act_candi, fsd_bak_candi, args)

        acm_loss_stack.append(loss.cpu().item())
        act_inst_loss_stack.append(loss_dict["act_inst_loss"])
        act_cont_loss_stack.append(loss_dict["act_cont_loss"])
        act_back_loss_stack.append(loss_dict["act_back_loss"])
        guide_loss_stack.append(loss_dict["guide_loss"])
        att_loss_stack.append(loss_dict["sparse_loss"])
        feat_loss_stack.append(loss_dict["feat_loss"])

        temp_cas = act_inst_cas

        test_tmp_data_log_dict[vid_name[0]] = {}
        test_tmp_data_log_dict[vid_name[0]]["vid_len"] = vid_len
        test_tmp_data_log_dict[vid_name[0]]["temp_att_score_np"] = temp_att.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_org_cls_score_np"] = act_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_ins_cls_score_np"] = act_inst_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_con_cls_score_np"] = act_cont_cas.cpu().numpy()
        test_tmp_data_log_dict[vid_name[0]]["temp_bak_cls_score_np"] = act_back_cas.cpu().numpy()

        fg_score = act_inst_cls[:, :args.action_cls_num]
        label_np = vid_label_t.cpu().numpy()
        score_np = fg_score.cpu().numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[score_np >= args.cls_threshold] = 1
        pred_np[score_np < args.cls_threshold] = 0
        correct_pred = np.sum(label_np == pred_np, axis=1)
        test_num_correct += np.sum((correct_pred == args.action_cls_num))
        test_num_total += correct_pred.shape[0]

        # GENERATE PROPORALS.
        temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
        temp_cls_score_np = np.reshape(temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_ins_score_np = temp_att[:, :, 0].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_con_score_np = temp_att[:, :, 1].unsqueeze(2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
        temp_att_ins_score_np = np.reshape(temp_att_ins_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
        temp_att_con_score_np = np.reshape(temp_att_con_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

        score_np = np.reshape(score_np, (-1))
        if score_np.max() > args.cls_threshold:
            cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0])
        else:
            cls_prediction = np.array([np.argmax(score_np)], dtype=np.int64)

        temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
        temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
        temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]

        test_tmp_data_log_dict[vid_name[0]]["temp_cls_score_np"] = temp_cls_score_np

        int_temp_cls_scores = upgrade_resolution(temp_cls_score_np, args.test_upgrade_scale)
        int_temp_att_ins_score_np = upgrade_resolution(temp_att_ins_score_np, args.test_upgrade_scale)
        int_temp_att_con_score_np = upgrade_resolution(temp_att_con_score_np, args.test_upgrade_scale)

        cas_act_thresh = np.arange(0.15, 0.25, 0.05)
        att_act_thresh = np.arange(0.15, 1.00, 0.05)

        proposal_dict = {}
        # CAS based proposal generation
        # cas_act_thresh = []
        for act_thresh in cas_act_thresh:

            tmp_int_cas = int_temp_cls_scores.copy()
            zero_location = np.where(tmp_int_cas < act_thresh)
            tmp_int_cas[zero_location] = 0

            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
                tmp_seg_list.append(pos)

            props_list = get_proposal_oic(tmp_seg_list, (1.0 * tmp_int_cas + 0.0 * int_temp_att_ins_score_np),
                                          cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)

            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += props_list[i]

        # att_act_thresh = []
        for att_thresh in att_act_thresh:

            tmp_int_att = int_temp_att_ins_score_np.copy()
            zero_location = np.where(tmp_int_att < att_thresh)
            tmp_int_att[zero_location] = 0

            tmp_seg_list = []
            for c_idx in range(len(cls_prediction)):
                pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
                tmp_seg_list.append(pos)

            props_list = get_proposal_oic(tmp_seg_list, (1.0 * int_temp_cls_scores + 0.0 * tmp_int_att), cls_prediction,
                                          score_np, t_factor, lamb=0.2, gamma=0.0)

            for i in range(len(props_list)):
                if len(props_list[i]) == 0:
                    continue
                class_id = props_list[i][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += props_list[i]

        # NMS
        final_proposals = []

        for class_id in proposal_dict.keys():
            final_proposals.append(nms(proposal_dict[class_id], args.nms_thresh))

        test_final_result['results'][vid_name[0]] = result2json(final_proposals, args.class_name_lst)

    test_acc = test_num_correct / test_num_total

    if args.test:
        # Final Test
        test_pred_txt_f = os.path.join(save_dir, "final_test_pred.txt")
        test_label_txt_f = os.path.join(save_dir, "final_test_label.txt")
        test_final_json_path = os.path.join(save_dir, "final_test_{}_result.json".format(args.dataset))
    else:
        # Train Evalutaion
        test_pred_txt_f = os.path.join(save_dir, "test_pred.txt")
        test_label_txt_f = os.path.join(save_dir, "test_label.txt")
        test_final_json_path = os.path.join(save_dir, "{}_lateset_result.json".format(args.dataset))

    np.savetxt(test_pred_txt_f, np.array(test_pred_score_stack), fmt="%.3f")
    np.savetxt(test_label_txt_f, np.array(test_vid_label_stack), fmt="%.3f")

    with open(test_final_json_path, 'w') as f:
        json.dump(test_final_result, f)

    anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                                   prediction_file=test_final_json_path,
                                   tiou_thresholds=args.tiou_thresholds,
                                   subset="test")

    test_mAP = anet_detection.evaluate()

    print("")
    print("test_act_inst_cls_loss:{:.3f}  test_act_cont_cls_loss:{:.3f}".format(np.mean(act_inst_loss_stack),
                                                                                np.mean(act_cont_loss_stack)))
    print("test_act_back_cls_loss:{:.3f}  test_att_loss:{:.3f}".format(np.mean(act_back_loss_stack),
                                                                       np.mean(att_loss_stack)))
    print(
        "test_feat_norm_loss:   {:.3f}  test_acm_loss:{:.3f}".format(np.mean(feat_loss_stack), np.mean(acm_loss_stack)))
    print("test acc:{:.3f}".format(test_acc))
    print("-------------------------------------------------------------------------------")

    test_log_dict = {"test_act_inst_cls_loss": np.mean(act_inst_loss_stack),
                     "test_act_cont_cls_loss": np.mean(act_cont_loss_stack),
                     "test_act_back_cls_loss": np.mean(act_back_loss_stack),
                     "test_feat_loss": np.mean(feat_loss_stack),
                     "test_att_loss": np.mean(att_loss_stack),
                     "test_acm_loss": np.mean(acm_loss_stack),
                     "test_acc": test_acc,
                     "test_mAP": test_mAP}

    return test_log_dict, test_tmp_data_log_dict