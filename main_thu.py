import os
import time
import pickle
import random
from tqdm import tqdm

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from config.model_config import build_args
from dataset.dataset_class import build_dataset, build_ftcl_dataset
from model.ACMNet import ACMNet
from model.FTCLNet import FTCLNet
from utils.net_utils import ACMLoss
from utils.ftcl_criterion import FTCLLoss

from train_thu import train
from test_thu import test


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda")

    if not args.test:
        save_dir = os.path.join(args.data_dir, "save", args.group, args.model_name)
    else:
        save_dir = os.path.dirname(args.checkpoint)

    args.save_dir = save_dir
    args.device = device

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if args.ftcl:
        model = FTCLNet(args)
    else:
        model = ACMNet(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model = model.to(device)

    if not args.test:
        if not args.without_wandb:
            wandb.init(name=time.asctime()[:-4] + args.model_name,
                       config=args,
                       group=args.group,
                       project=f"FTCL_{args.dataset}",
                       sync_tensorboard=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        #                              betas=(0.9, 0.999), weight_decay=args.weight_decay)

        train_dataset = build_dataset(args, phase="train", sample="random")
        test_dataset = build_dataset(args, phase="test", sample="uniform")

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=False)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, drop_last=False)

        ftcl_dataset = build_ftcl_dataset(args, phase="train", sample="random")
        ftcl_dataloader = DataLoader(ftcl_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=False)
        if args.ftcl:
            criterion = FTCLLoss(args)
        else:
            criterion = ACMLoss(args)

        best_test_mAP = 0

        for epoch_idx in tqdm(range(args.start_epoch, args.epochs)):

            train_log_dict = train(args, model, train_dataloader, ftcl_dataloader, criterion, optimizer)

            if epoch_idx >= args.start_test_epoch:
                with torch.no_grad():
                    test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
                    test_mAP = test_log_dict["test_mAP"]

                if test_mAP > best_test_mAP:
                    best_test_mAP = test_mAP
                    checkpoint_file = f"{args.dataset}_best.pth"
                    torch.save({
                        'epoch': epoch_idx,
                        'model_state_dict': model.state_dict()
                    }, os.path.join(save_dir, checkpoint_file))

                    with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                        pickle.dump(test_tmp_data_log_dict, f)

                checkpoint_file = f"{args.dataset}_latest.pth"
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict()
                }, os.path.join(save_dir, checkpoint_file))

                print("Current test_mAP:{:.4f}, Current Best test_mAP:{:.4f} Current Epoch:{}/{}".format(test_mAP,
                                                                                                         best_test_mAP,
                                                                                                         epoch_idx,
                                                                                                         args.epochs))
                print("-------------------------------------------------------------------------------")

                if not args.without_wandb:
                    wandb.log(train_log_dict)
                    wandb.log(test_log_dict)
                    wandb.log({"best_test_mAP": best_test_mAP})

    else:
        test_dataset = build_dataset(args, phase="test", sample="uniform")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, drop_last=False)
        criterion = ACMLoss(args)

        with torch.no_grad():
            test_log_dict, test_tmp_data_log_dict = test(args, model, test_dataloader, criterion)
            test_mAP = test_log_dict["test_mAP"]

            with open(os.path.join(save_dir, "test_tmp_data_log_dict.pickle"), "wb") as f:
                pickle.dump(test_tmp_data_log_dict, f)


if __name__ == "__main__":
    args = build_args(dataset="THUMOS")
    setup_seed(args.seed)
    print(args)
    main(args)
