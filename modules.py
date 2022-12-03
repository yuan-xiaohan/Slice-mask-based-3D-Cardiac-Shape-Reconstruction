from dataset import *
from metrics import *
import os
from torch.autograd import Variable


def train_model(models, data_train, losses, opts, schs):
    """Train the model and report validation error with training error
    """
    total_dice = 0
    total_loss1 = 0
    total_loss2 = 0
    total_tol_loss = 0
    model1 = models[0]
    model2 = models[1]
    loss_fun1 = losses[0]
    loss_fun2 = losses[1]
    opt1 = opts[0]
    opt2 = opts[1]
    sch1 = schs[0]
    sch2 = schs[1]
    model1.train()
    model2.train()
    for batch_id, (images, label) in enumerate(data_train):
        bs = images.shape[0]
        masks = label[0]
        pca = label[1]
        images = images.view(-1, 1, 192, 192)
        masks = masks.view(-1, 192, 192)

        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        pca = Variable(pca.cuda())

        # seg model
        output1 = model1(images)
        loss1 = loss_fun1(output1, masks)

        # para model
        output1 = torch.softmax(output1, dim=1)
        pred1 = output1
        output1 = torch.argmax(output1, dim=1).float()
        output1 = output1.view(bs, -1, 192, 192)
        output2 = model2(output1)
        loss2 = loss_fun2(output2, pca)

        # update
        tol_loss = loss1 + 10 * loss2
        opt1.zero_grad()
        opt2.zero_grad()
        tol_loss.backward()
        opt1.step()
        opt2.step()
        sch1.step()
        sch2.step()

        dice = dice_check_for_batch(pred1[:, 1, :, :].cpu(), masks.cpu().float(),
                                     images.size()[0])
        total_dice = total_dice + dice
        total_loss1 = total_loss1 + loss1.cpu().item()
        total_loss2 = total_loss2 + loss2.cpu().item()
        total_tol_loss = total_tol_loss + tol_loss.cpu().item()
    loss_list = [total_loss1/ (batch_id + 1), total_loss2/ (batch_id + 1), total_tol_loss/ (batch_id + 1)]
    return total_dice / (batch_id + 1), loss_list


def test_model(models, data_test, losses):
    """
        Test run
    """
    # calculating validation loss
    total_dice = 0
    total_loss1 = 0
    total_loss2 = 0
    total_tol_loss = 0
    model1 = models[0]
    model2 = models[1]
    loss_fun1 = losses[0]
    loss_fun2 = losses[1]
    out_pca_list = []
    pred_list = []
    for batch_id, (images, label) in enumerate(data_test):
        with torch.no_grad():
            bs = images.shape[0]
            masks = label[0]
            pca = label[1]
            images = images.view(-1, 1, 192, 192)
            masks = masks.view(-1, 192, 192)

            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            pca = Variable(pca.cuda())

            # seg model
            output1 = model1(images)
            loss1 = loss_fun1(output1, masks)

            # para model
            output1 = torch.softmax(output1, dim=1)
            pred1 = output1
            output1 = torch.argmax(output1, dim=1).float()
            output1 = output1.view(bs, -1, 192, 192)
            output2 = model2(output1)
            loss2 = loss_fun2(output2, pca)
            tol_loss = loss2 + loss1

            # compute loss and accuracy
            dice = dice_check_for_batch(pred1[:, 1, :, :].cpu(), masks.cpu().float(),
                                        images.size()[0])
            total_dice = total_dice + dice
            total_loss1 = total_loss1 + loss1.cpu().item()
            total_loss2 = total_loss2 + loss2.cpu().item()
            total_tol_loss = total_tol_loss + tol_loss.cpu().item()

            # return outputs
            pred1 = torch.argmax(pred1, dim=1).float().cpu().data.numpy()
            pred_list.append(pred1)
            out_pca_list.append(output2[0, :].cpu().numpy())

    loss_list = [total_loss1 / (batch_id + 1), total_loss2 / (batch_id + 1), total_tol_loss / (batch_id + 1)]
    return total_dice / (batch_id + 1), loss_list, np.array(pred_list), np.array(out_pca_list)


def save_prediction_image(preds, batch_id, save_folder_name):
    """save images to save_path
    """
    # organize images in every epoch
    desired_path = os.path.join(save_folder_name, "%03d" % batch_id)
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    for i in range(preds.shape[0]):
        img = preds[i, :, :] * 255
        img_np = img.astype('uint8')

        # Save Image
        export_name = "%03d"%i + '.png'
        cv2.imwrite(os.path.join(desired_path, export_name), img_np)


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img