import torch.nn as nn
from networks.unet import UNet
from networks.para_model import para_model
from modules import *
from processing import *


PCA_NUM = 51
input_ch_num = 13

class weighted_mse_loss_WITHSCALE(nn.Module):
    def __init__(self, weights):
        super(weighted_mse_loss_WITHSCALE, self).__init__()
        self.weights = torch.from_numpy(weights).float().cuda()

    def forward(self, out, label):
        weights = self.weights.unsqueeze(dim=0)
        weights = torch.cat([weights, torch.FloatTensor([[1.]]).cuda()], dim=1)
        pct_var = (out - label)**2
        output = pct_var * weights
        loss = output.mean()
        return loss


def main_function(input_path_train, save_path, pca_info, pretrained_path):
    # Saving images and models directories
    save_path_model = os.path.join(save_path, "models")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_model):
        os.mkdir(save_path_model)

    aver_v, face, VT, var_ratio = PCA_operation(pca_info)
    weights = var_ratio

    # Dataset begin
    train = DataTrain(input_path_train)

    # Dataloader begin
    train_load = \
        torch.utils.data.DataLoader(dataset=train,
                                    num_workers=0, batch_size=4, shuffle=True)

    # Model
    model1 = UNet(in_channels=1, out_channels=2).cuda()
    model2 = para_model(input_ch_num, PCA_NUM).cuda()
    if len(pretrained_path) != 0:
        model2.load_state_dict(torch.load(pretrained_path))
    models = [model1, model2]

    # Loss function
    loss1 = nn.CrossEntropyLoss()
    loss2 = weighted_mse_loss_WITHSCALE(weights)
    losses = [loss1, loss2]

    # Optimizerd
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    opts = [opt1, opt2]

    sch1 = torch.optim.lr_scheduler.MultiStepLR(opt1, milestones=[20, 50], gamma=0.9)
    sch2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=[20, 50], gamma=0.9)
    schs = [sch1, sch2]

    # Parameters
    epoch_start = 0
    epoch_end = 100
    # Train
    print("Initializing Training!")
    for epoch in range(epoch_start, epoch_end):
        # train the model
        total_dice, loss_list = train_model(models, train_load, losses, opts, schs)
        print('TRAIN EPOCH:{},loss1={:.6f},loss2={:.6f},total loss={:.6f}'.format(epoch + 1, loss_list[0], loss_list[1],
                                                                                  loss_list[2]))

        if (epoch + 1) % 100 == 0:
            torch.save(model1.state_dict(), os.path.join(save_path_model, r"seg_epoch_{0}.pkl".format(epoch + 1)),
                       _use_new_zipfile_serialization=False)
            torch.save(model2.state_dict(), os.path.join(save_path_model, r"para_epoch_{0}.pkl".format(epoch + 1)),
                       _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    import argparse
    import sys

    arg_parser = argparse.ArgumentParser(description="Train the model jointly .")
    arg_parser.add_argument(
        "--train",
        "-t",
        dest="train_directory",
        required=True,
        help="The training data directory. "
    )
    arg_parser.add_argument(
        "--save",
        "-s",
        dest="save_directory",
        required=True,
        help="The save directory.",
    )
    arg_parser.add_argument(
        "--info",
        "-i",
        dest="pca_info_directory",
        required=True,
        help="The pca info directory.",
    )
    arg_parser.add_argument(
        "--pretrained",
        "-pre",
        dest="pretrained_directory",
        default="",
        help="The pre-trained parameter regression model directory.",
    )

    # sys.argv = [r"train_main.py",
    #             "--train", r"SliceMask_LVCT\train",
    #             "--save", r"History",
    #             "--info", r"info",
    #             "--pretrained", "models/pretrained_para.pkl"
    #             ]
    args = arg_parser.parse_args()
    main_function(args.train_directory, args.save_directory, args.pca_info_directory, args.pretrained_directory)
