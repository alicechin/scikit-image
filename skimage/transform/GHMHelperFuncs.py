import os
import numpy as np
import matplotlib.pyplot as plt

#GENERAL IMAGES HELPER FUNCTIONS
def is_grayscale(img):
    if len(img.shape) == 2:
        return True
    c1 = img[:,:,0]
    c2 = img[:,:,1]
    c3 = img[:,:,2]
    return np.array_equal(c1,c2) and np.array_equal(c2,c3)

def plot_hist(img):
    plt.hist(img.ravel(), range=(0,256), bins=256)
    plt.show()

# if all pixels are the same value, a "white" image will look black, since min=max and normally min=black and max=white so it's ambiguous
def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.show()

def is_from_0_to_255(img):
    return np.all(img <= 255) and np.all(img >= 0)

def read_and_check_img(filepath):
    if filepath[-4:] != ".jpg":
        raise Exception("Expect a .jpg image. Received something else.")
    print("Reading image " + filepath)
    img = plt.imread(filepath)
    if not is_grayscale(img):
        raise Exception("Image is not a grayscale image (meaning not all 3 channels of jpg are the same.)")
    if len(img.shape) != 2:
        img = img[:,:,0]
    show_img(img)
    plot_hist(img)
    print("Done reading "+ filepath)
    return img

# TODO check M cdf (def function and use it)
# TODO check C cdf (def function and use it)
# TODO check T cdf (def function and use it)

# VERIFICATION FUNCTIONS
def check_M_matrix_pdf(M):
    # all elements should be either 0 or 1
    if not np.logical_or(M==0, M==1):
        return False
    # every column should have exactly one 1
    if not np.all(np.sum(M, axis=0) == 1):
        return False
    # get indices for all nonzero (1) elements in M
    row_indices, col_indices = np.nonzero(M)
    m, n = M.shape
    for j in range(n):
        c = col_indices[j]
        # check the top-left to bottom-right structure
        if (c != j):
            return False
    return True

def check_C_matrix_pdf(C):
    # left to right: cost increases
    if C != np.sort(C, axis=1):
        return False
    # top to bottom: cost decreases
    if C != -np.sort(-C, axis=0):
        return False
    return True

def check_T_matrix_pdf(T):
    m, n = T.shape
    for i in range(m):
        for j in range(n):
            if (not np.issubdtype(T[i,u], np.intenger)) or T[i,j] < -1 or T[i,j] > j:
                return False

#GHM HELPER FUNCTIONS
# any distance must be 'additive' (see paper)
def distance(value1, value2, metric='L1'):
    if metric == 'L1':
        return abs(value2-value1)
    elif metric == 'L2':
        return (value2-value1)**2
    else:
        raise Exception("Currently only accepts L1 squared and L2 squared distance metrics.")

# used in find_mapping
def row_cost(A, B, i, jj, j, dist='L1'):
    """See paper for rowCost function. Note that j and k are inclusive."""
    dist_sum = 0
    (n, k) = A.shape
    for col in range(k):
        dist_sum += distance(sum(A[jj:j+1, col]), B[i, col], dist)
    return dist_sum

# used in create_matrix
def calc_histogram(img):
#     vals, indices, counts = np.unique(img.ravel(), return_inverse=True, return_counts=True)
#     counts = counts / img.size    
#     return np.array([counts]).T, vals
    histogram = [0 for _ in range(256)]
    (h,w) = img.shape
    for i in range(h):
        for j in range(w):
            histogram[img[i,j]] += 1
    histogram = np.array([histogram])
    histogram = histogram / img.size
    return histogram.T

def calc_cdf(mtx):
    n = mtx.shape[0]
    H = np.tril(np.ones((n, n)))
    return np.matmul(H, mtx)

def create_matrix(img, num_histograms_per_dim=1):
    (height, width) = img.shape
    matrix = None
    box_height = height // num_histograms_per_dim
    box_width = width // num_histograms_per_dim
    for i in range(num_histograms_per_dim-1):
        for j in range(num_histograms_per_dim-1):
            top = i*box_height
            bottom = (i+1)*box_height
            left = j*box_width
            right = (j+1)*box_width
            sub_img = img[top:bottom, left:right]
            show_img(sub_img)
            column = calc_histogram(sub_img)
            if matrix is None:
                matrix = column
            else:
                matrix = np.hstack((matrix, column))
    # last boxes:
    # on the bottom:
    i = num_histograms_per_dim - 1
    top = i*box_height
    for j in range(num_histograms_per_dim-1):
        left = j*box_width
        right = (j+1)*box_width
        sub_img = img[top:, left:right]
        show_img(sub_img)
        column = calc_histogram(sub_img)
        if matrix is None:
            matrix = column
        else:
            matrix = np.hstack((matrix, column))
    # on the right:
    j = num_histograms_per_dim - 1
    left = j*box_width
    for i in range(num_histograms_per_dim-1):
        top = i*box_height
        bottom = (i+1)*box_height
        sub_img = img[top:bottom, left:]
        show_img(sub_img)
        column = calc_histogram(sub_img)
        if matrix is None:
            matrix = column
        else:
            matrix = np.hstack((matrix, column))
    # bottom-right box
    i = num_histograms_per_dim - 1
    j = num_histograms_per_dim - 1
    top = i*box_height
    left = j*box_width
    sub_img = img[top:, left:]
    show_img(sub_img)
    column = calc_histogram(sub_img)
    if matrix is None:
        matrix = column
    else:
        matrix = np.hstack((matrix, column))
    
#     matrix = calc_histogram(img)
    return matrix

def convert(imgA, mapping):
    converter = np.vectorize(lambda pix : mapping[pix])
    return converter(imgA)



########################## 3D functions ##########################

# def create_matrix3D(img, num_histograms_per_dim=1):
#     """Creates the matrices using histograms as columns (see paper)."""
#     (height, width, depth) = img.shape
#     box_height = height // num_histograms_per_dim
#     box_width = width // num_histograms_per_dim
#     box_depth = depth // num_histograms_per_dim

#     pix_to_index, index_to_pix = pix_index_mapping(img)

#     matrix = None
#     for i in range(num_histograms_per_dim-1):
#         for j in range(num_histograms_per_dim-1):
#             for k in range(num_histograms_per_dim-1):
#                 top = i*box_height
#                 bottom = (i+1)*box_height
#                 left = j*box_width
#                 right = (j+1)*box_width
#                 front = k*box_depth
#                 back = (k+1)*box_depth
#                 sub_img = img[top:bottom, left:right, front:back]
#                 show_slices(sub_img)
#                 column = calc_histogram(sub_img, pix_to_index)
#                 if matrix is None:
#                     matrix = column
#                 else:
#                     matrix = np.hstack((matrix, column))

#     # last boxes:
#     for k in range(num_histograms_per_dim-1): # for each slice except last
#         front = k*box_depth
#         back = (k+1)*box_depth
#         # on the bottom:
#         i = num_histograms_per_dim - 1
#         top = i*box_height
#         for j in range(num_histograms_per_dim-1):
#             left = j*box_width
#             right = (j+1)*box_width
#             sub_img = img[top:, left:right, front:back]
#             show_slices(sub_img)
#             column = calc_histogram(sub_img, pix_to_index)
#             if matrix is None:
#                 matrix = column
#             else:
#                 matrix = np.hstack((matrix, column))
#         # on the right:
#         j = num_histograms_per_dim - 1
#         left = j*box_width
#         for i in range(num_histograms_per_dim-1):
#             top = i*box_height
#             bottom = (i+1)*box_height
#             sub_img = img[top:bottom, left:, front:back]
#             show_slices(sub_img)
#             column = calc_histogram(sub_img, pix_to_index)
#             if matrix is None:
#                 matrix = column
#             else:
#                 matrix = np.hstack((matrix, column))
#         # bottom-right box
#         i = num_histograms_per_dim - 1
#         j = num_histograms_per_dim - 1
#         top = i*box_height
#         left = j*box_width
#         sub_img = img[top:, left:, front:back]
#         show_slices(sub_img)
#         column = calc_histogram(sub_img, pix_to_index)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))

#     # very back slice
#     k = num_histograms_per_dim - 1
#     front = k*box_depth
#     # on the bottom:
#     i = num_histograms_per_dim - 1
#     top = i*box_height
#     for j in range(num_histograms_per_dim-1):
#         left = j*box_width
#         right = (j+1)*box_width
#         sub_img = img[top:, left:right, front:]
#         show_slices(sub_img, pix_to_index)
#         column = calc_histogram(sub_img)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))
#     # on the right:
#     j = num_histograms_per_dim - 1
#     left = j*box_width
#     for i in range(num_histograms_per_dim-1):
#         top = i*box_height
#         bottom = (i+1)*box_height
#         sub_img = img[top:bottom, left:, front:]
#         show_slices(sub_img)
#         column = calc_histogram(sub_img, pix_to_index)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))
#     # bottom-right box
#     i = num_histograms_per_dim - 1
#     j = num_histograms_per_dim - 1
#     top = i*box_height
#     left = j*box_width
#     sub_img = img[top:, left:, front:]
#     show_slices(sub_img)
#     column = calc_histogram(sub_img, pix_to_index)
#     if matrix is None:
#         matrix = column
#     else:
#         matrix = np.hstack((matrix, column))
#     return matrix, pix_to_index, index_to_pix
