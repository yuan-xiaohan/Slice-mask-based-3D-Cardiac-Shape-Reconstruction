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

# compute points error
def points_error(gt_path, test_path):
    gt = o3d.io.read_triangle_mesh(gt_path)
    test = o3d.io.read_triangle_mesh(test_path)
    # transform obj to pcd
    pcd_gt = o3d.geometry.PointCloud()  # create an empty geometry
    pcd_gt.points = gt.vertices
    pcd_test = o3d.geometry.PointCloud()
    pcd_test.points = test.vertices

    dists_t2g = pcd_test.compute_point_cloud_distance(pcd_gt)
    msd_t2g = np.asarray(dists_t2g)

    dists_g2t = pcd_gt.compute_point_cloud_distance(pcd_test)
    msd_g2t = np.asarray(dists_g2t)

    cd = msd_t2g.mean() + msd_g2t.mean()

    return msd_t2g.mean(), cd


