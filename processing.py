from obj_process import *
PCA_NUM = 51

def normalize_get_ms(pca_list):
    # get mean and sigma from data and return normalized data
    mean = pca_list.mean(axis=0)
    sigma = np.sqrt(np.var(pca_list, axis=0))
    pca_new = (pca_list - np.expand_dims(mean, 0).repeat(pca_list.shape[0], axis=0)) \
              / np.expand_dims(sigma, 0).repeat(pca_list.shape[0], axis=0)

    return pca_new, mean, sigma


def normalize(pca_list, mean, sigma):
    # return normalized data
    # pca_list: [N, PCA_NUM]
    pca_new = (pca_list - np.expand_dims(mean, 0).repeat(pca_list.shape[0], axis=0)) \
              / np.expand_dims(sigma, 0).repeat(pca_list.shape[0], axis=0)
    return pca_new


def inverse_normalize(pca_new, mean, sigma):
    pca_list = pca_new * np.expand_dims(sigma, 0).repeat(pca_new.shape[0], axis=0) + \
               np.expand_dims(mean, 0).repeat(pca_new.shape[0], axis=0)

    return pca_list


def PCA_operation(file_base):
    """
    Output：
    # X_aver: (v_num, 3)
    # X_new: (v_num*3, k)
    # VT: (k, n)
    # X_pca_ratio: (k, )
    """

    aver_dir = os.path.join(file_base, "aver.obj")
    VT_dir = os.path.join(file_base, "VT.txt")
    var_ratio_dir = os.path.join(file_base, "pca_ratio.txt")

    # get mean obj
    aver_v, face = obj_read(aver_dir)
    VT = np.loadtxt(VT_dir, dtype=np.float)
    var_ratio = np.loadtxt(var_ratio_dir, dtype=np.float)

    return aver_v, face, VT, var_ratio


def reconstruction(pca_coff, VT, aver_v):
    """
    # VT: (k, n), n = v_num * 3
    # pca_coff: (m, k)
    # aver: (v_num, 3)
    # m_reconstruct:(m, v_num, 3)
    """
    m_reconstruct = np.dot(pca_coff, VT)  # X = X_new * VT, (m, v_num * 3)
    m_reconstruct = np.repeat(np.expand_dims(aver_v, axis=0), pca_coff.shape[0], axis=0) + np.reshape(m_reconstruct,
                                                                                                      [pca_coff.shape[0],
                                                                                                       aver_v.shape[0],
                                                                                                       aver_v.shape[1]])
    return m_reconstruct

def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

def normalization1(image, mean, std):
    """ Normalization using mean and std
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """

    image = image / 255  # values will lie between 0 and 1.
    image = (image - mean) / std

    return image


def normalization2(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    return image_new