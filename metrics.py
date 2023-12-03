# dice coefficient
def dice_check(mask, prediction):

    smooth = 1.
    m1 = mask.flatten()  # Flatten
    m2 = prediction.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    return dice


def dice_check_for_batch(masks, predictions, batch_size):
    total_dice = 0
    for index in range(batch_size):
        total_dice += dice_check(masks[index], predictions[index])

    return total_dice/batch_size


# compute seg Iou
def iou_check(mask, prediction):
    smooth = 1.
    m1 = mask.flatten()  # Flatten
    m2 = prediction.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    union = m1.sum() + m2.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def iou_check_for_batch(masks, predictions, batch_size):
    total_iou = 0
    for index in range(batch_size):
        total_iou += iou_check(masks[index], predictions[index])
    return total_iou / batch_size


