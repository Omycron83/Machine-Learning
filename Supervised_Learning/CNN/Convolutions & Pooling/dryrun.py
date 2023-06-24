import numpy as np
import time
import numba
from skimage.util.shape import view_as_windows
@numba.jit(forceobj = True)
def padding(pic, padX, padY):
    return np.pad(pic, (padX, padY), constant_values=(0, 0))

def avg_pooling(input):
    return np.average(input)

def max_pooling(input):
    return np.max(input)

def pooling(pic, pool, poolX, poolY, stride):
    pool_pic = np.zeros((int((pic.shape[0] - poolX) / stride + 1), int((pic.shape[1] - poolY) / stride + 1)))
    i_pool = 0
    for i in range(0, pic.shape[1] + 1 - poolY, stride):
        j_pool = 0
        for j in range(0, pic.shape[0] + 1 - poolX, stride):
            pool_pic[j_pool, i_pool] = pool(pic[j:j + poolX, i:i + poolY])
            j_pool += 1
        i_pool += 1
    return pool_pic

#"Naive" implementation
@numba.jit(forceobj= True)
def convolution(pic, filter, padX, padY, stride):
    conv_pic = np.zeros((int((pic.shape[0] + 2*padX - filter.shape[0]) / stride + 1), int((pic.shape[1] + 2*padY - filter.shape[1]) / stride + 1)))                                                                                                                   
    pic = padding(pic, padX, padY)
    i_conv = 0
    for i in range(0, pic.shape[1] - filter.shape[1] + 1, stride):
        j_conv = 0
        for j in range(0, pic.shape[0] - filter.shape[0] + 1, stride):
            conv_pic[j_conv, i_conv] = np.sum(pic[j:j + filter.shape[0], i:i + filter.shape[1]] * filter)
            j_conv += 1
        i_conv += 1
    return conv_pic

@numba.jit(forceobj = True)
def im2col(pic, filter, stride):
    im2col_matrix = np.zeros((filter.size, int((pic.shape[0] - filter.shape[0]) / stride + 1) * int((pic.shape[1] - filter.shape[1]) / stride + 1)))
    col = 0
    for i in range(0, pic.shape[1] - filter.shape[1] + 1, stride):
        for j in range(0, pic.shape[0] - filter.shape[0] + 1, stride):
            im2col_matrix[:, col] = pic[j:j + filter.shape[0], i:i + filter.shape[1]].reshape(1, filter.size)
            col += 1
    
    return im2col_matrix

#Vectorized + striding window
@numba.jit(forceobj = True)
def memory_strided_im2col(pic, filter, padX, padY, stride):
    pic = padding(pic, padX, padY)
    im2col_matrix = view_as_windows(pic, filter.shape, step = stride).reshape(filter.size, int((pic.shape[0] - filter.shape[0]) / stride + 1) * int((pic.shape[1] - filter.shape[1]) / stride + 1))
    return (filter.flatten() @ im2col_matrix).reshape(int((pic.shape[0] + 2*padX - filter.shape[0]) / stride + 1), int((pic.shape[1] + 2*padY - filter.shape[1]) / stride + 1)) 

#Trying my own stuff
def view_as_window(pic, filter, stride):
    #This implementation uses 
    if pic.dtype != np.float64:
        pic = pic.astype(np.float64)
    print(pic.strides)
    print(pic.shape)
    pic_windowed = np.lib.stride_tricks.as_strided(pic, shape = (filter.size, int((pic.shape[0] - filter.shape[0]) / stride + 1) * int((pic.shape[1] - filter.shape[1]) / stride + 1)), strides = (4 * stride, filter.shape[1] * 4, 4))
    return pic_windowed.reshape()

#view_as_window(np.array([[5, 4], [3, 2]]), 1, 2)
#Vectorized implementation
@numba.jit(forceobj = True) 
def conv_im2col(pic, filter, padX, padY, stride):
    pic_pooled = padding(pic, padX, padY)
    conv_matrix = im2col(pic_pooled, filter, stride)
    conv_pic = (filter.flatten() @ conv_matrix).reshape(int((pic.shape[0] + 2*padX - filter.shape[0]) / stride + 1), int((pic.shape[1] + 2*padY - filter.shape[1]) / stride + 1))
    return conv_pic

def unit_pooling_conv():
    amt = 24000
    array = np.arange(10000).reshape(100, 100)
    filter = np.array(([[2, 2], [2, 2]])) 
    """
    x = time.time()
    for i in range(amt):
        if i % 1000 == 0:
            print(i)
        convolution(array, filter, 0, 0, 1)
    print(time.time() - x)
    
    x = time.time()
    for i in range(amt):
        if i % 1000 == 0:
            print(i)
        conv_im2col(array, filter, 0, 0, 1)
    print(time.time() - x)
    """
    memory_strided_im2col(array, filter, 0, 0, 1)
    x = time.time()
    for i in range(amt):
        if i % 1000 == 0:
            print(i)
        memory_strided_im2col(array, filter, 0, 0, 1)
    print(time.time() - x)

def im2col_test():
    mat = np.array([[3, 9, 0], [2, 8, 1], [1, 4, 8]])
    filters = np.array([[8, 9], [4, 4]])
    print(convolution(mat, filters, 0, 0, 1))
    print(conv_im2col(mat, filters, 0, 0, 1))
    print(memory_strided_im2col(mat, filters, 0, 0, 1))

def sobel_filter_number():
    import matplotlib.pyplot as plt
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    pic = train_X[0].reshape(28, 28)
    plt.imshow(pic, cmap = 'gray', vmax = 255, vmin = 0)
    plt.show()
    sobel_vert = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_hori = np.array([[-1, -2, -1], [0,0,0], [1,2,1]])

    pic_conv_vert = convolution(pic, sobel_vert, 1, 1, 1)
    pic_conv_vert = pooling(pic_conv_vert, avg_pooling, 2, 2, 2)

    pic_conv_hori = convolution(pic, sobel_hori, 1, 1, 1)
    pic_conv_hori = pooling(pic_conv_hori, avg_pooling, 2, 2, 2)
    plt.imshow(pic_conv_vert, cmap = 'gray', vmax = 255, vmin = 0)
    plt.show()
    plt.imshow(pic_conv_hori, cmap = 'gray', vmax = 255, vmin = 0)
    plt.show()

unit_pooling_conv()