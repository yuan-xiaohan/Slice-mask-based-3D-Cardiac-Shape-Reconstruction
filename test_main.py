import torch.nn as nn
from networks.unet import UNet
from networks.para_model import para_model
from modules import *
from processing import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def Test(models, data_test):
    """
        Test run
    """
    # calculating validation loss
    model1 = models[0]
    model2 = models[1]
    out_pca_list = []
    pred_list = []
    for batch_id, (images) in enumerate(data_test):
        with torch.no_grad():
            bs = images.shape[0]
            images = images.view(-1, 1, 192, 192)
            images = Variable(images.cuda())
            # seg model
            output1 = model1(images)

            # para model
            output1 = torch.softmax(output1, dim=1)
            pred1 = output1
            output1 = torch.argmax(output1, dim=1).float()
            output1 = output1.view(bs, -1, 192, 192)
            output2 = model2(output1)

            # return outputs
            pred1 = torch.argmax(pred1, dim=1).float().cpu().data.numpy()
            pred_list.append(pred1)
            out_pca_list.append(output2[0, :].cpu().numpy())
    return np.array(pred_list), np.array(out_pca_list)


def main_function(input_path_test, save_path, pca_info):
    model1_path = os.path.join(save_path, "models", "seg.pkl")
    model2_path = os.path.join(save_path, "models", "para.pkl")

    # Saving images and models directories
    save_path_pred = os.path.join(save_path, "pred_masks")
    save_path_obj = os.path.join(save_path, "pred_shape")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_pred):
        os.mkdir(save_path_pred)
    if not os.path.exists(save_path_obj):
        os.mkdir(save_path_obj)

    aver_v, face, VT, var_ratio = PCA_operation(pca_info)
    nor_list = np.loadtxt(os.path.join(pca_info, "nor_list.txt"), dtype=np.float32)
    mean = nor_list[0, :]
    sigma = nor_list[1, :]
    # Dataset begin
    test = OnlyTest(input_path_test)

    # Dataloader begin
    test_load = \
        torch.utils.data.DataLoader(dataset=test,
                                    num_workers=0, batch_size=1, shuffle=False)

    # Model
    model1 = UNet(in_channels=1, out_channels=2).cuda()
    model2 = para_model(input_ch_num, PCA_NUM).cuda()
    model1.load_state_dict(torch.load(model1_path))
    model2.load_state_dict(torch.load(model2_path))
    models = [model1, model2]

    # Test
    mask_out, pca_out = Test(models, test_load)
    # visualize obj
    theta = pca_out * sigma + mean
    if theta.shape == tuple([theta.shape[0], ]):
        theta = np.reshape(theta, [1, theta.shape[0]])
    v = reconstruction(theta[:, 0:PCA_NUM - 1], VT, aver_v)
    for i in range(v.shape[0]):
        save_prediction_image(mask_out[i, :], i, save_path_pred)
        v_s = v[i, :] * theta[i, PCA_NUM - 1]
        obj_write(os.path.join(save_path_obj, "%03d" % i + ".obj"), v_s, face)


if __name__ == "__main__":
    import argparse
    import sys

    arg_parser = argparse.ArgumentParser(description="Test the model.")
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_directory",
        required=True,
        help="The testing data directory. "
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

    # sys.argv = [r"test_main.py",
    #             "--test", r"SliceMask_LVCT\test",
    #             "--save", r"History",
    #             "--info", r"info"
    #             ]
    args = arg_parser.parse_args()
    main_function(args.test_directory, args.save_directory, args.pca_info_directory)