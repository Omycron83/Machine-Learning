import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba import cuda
import winsound
from keras.datasets import mnist
#building a gan, but we arent using the wasserstein loss because as the ordinary loss just sucks (vanishing gradient) i dont know how the gradient is formed and therefore RIP


@njit
def con(x):
    return np.ascontiguousarray(x)

@njit
def rounds(x):
    out = np.empty_like(x)
    return np.round(x, 0, out)

#note: using leakyReLU instead of regular ReLU in order to prevent 'dead units' and too much information being lost
@njit
def ReLU(Z):
    return (Z > 0) * Z + (Z < 0) * Z * 0.1

@njit
def ReLUDeriv(Z):
    return (Z > 0) + (Z < 0) * 0.1

@njit(parallel=True)
def sigmoidDeriv(i):
    return sigmoid(i) * (1 - sigmoid(i))

@njit(parallel=True)
def sigmoid(i):
    sig = 1/(1 + np.exp(-i))
    sig = np.minimum(sig, 0.9999999)
    sig = np.maximum(sig, 0.0000001)
    return sig

def nn_runner(train_X, train_Y, iterations, _lambda, alpha, alpha_gen, batchsize):
    accuracy_list = []
    temp = []
    epoch_list = []
    hidden1 = 30
    #chances for dropout as we want to prevent overfitting
    q = 0.5
    hidden2 = 30
    
    theta1 = np.random.rand(hidden1,np.shape(train_X)[1] + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta1)[1]))
    theta1 = theta1 * 2 * e - e

    theta2 = np.random.rand(hidden2,hidden1 + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta2)[1]))
    theta2 = theta2 * 2 * e - e

    theta3 = np.random.rand(1, hidden2 + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta3)[1]))
    theta3 = theta3 * 2 * e - e

    noise_shape = 20
    hidden1_gan = 40
    hidden2_gan = 120

    theta1_gan = np.random.rand(hidden1_gan, noise_shape + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta1_gan)[1]))
    theta1_gan = theta1_gan * 2 * e - e

    theta2_gan = np.random.rand(hidden2_gan,hidden1_gan + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta2_gan)[1]))
    theta2_gan = theta2_gan * 2 * e - e

    theta3_gan = np.random.rand(784, hidden2_gan + 1)
    e = np.sqrt(2) / (np.sqrt(np.shape(theta3_gan)[1]))
    theta3_gan = theta3_gan * 2 * e - e
    
    const1 = 400
    const2 = const1 * alpha
    gan_history = np.random.randn(784,)
    print("Hi")
    for i in range(1,iterations+1):
        #slowly decreasing the learning rate with every iteration to "guarantee" convergence
        alpha = (const2/(const1 + i))
        averageAccuracy = 0
        averageError = 0
        p = np.random.permutation(len(train_Y))
        batch = train_X[p]
        batchY = train_Y[p]
        theta1_gan_grad, theta2_gan_grad, theta3_gan_grad = 0, 0, 0
        for j in range(len(train_Y)//batchsize + 1):
            if len(batchY) > batchsize:
                #
                noise = np.random.randn(batch[-batchsize:, :].shape[0], noise_shape)
                z1_gen, z2_gen, a1_gen, a2_gen, images = generatorForward(noise, theta1_gan, theta2_gan, theta3_gan, q)
                currSize = batchY[-batchsize:].shape[0]
                curr_batch = np.vstack((batch[-batchsize:, :], images))            
                curr_batchY = np.hstack((np.random.randint(0, 100, (images.shape[0])) * 0.001, np.random.randint(900, 1000, (images.shape[0])) * 0.001)).reshape(batchY[-batchsize:].shape[0] * 2, 1)
                #addings instance noise in order to help make the distributions overlapp more and lead to better convergence
                noiseLevel = 0.3 #0.3
                curr_batch += np.random.randn(curr_batch.shape[0], curr_batch.shape[1]) * noiseLevel
                #
                J, accuracy, predValues, z2, z1, a2, a1 = discriminatorForward(curr_batch, curr_batchY, _lambda, theta1, theta2, theta3, q)
                batch, batchY = batch[:-batchsize], batchY[:-batchsize]
                         
            else:
                #
                currSize = batchY[-batchsize:].shape[0]
                noise = np.random.randn(batch[-batchsize:, :].shape[0], noise_shape)
                z1_gen, z2_gen, a1_gen, a2_gen, images = generatorForward(noise, theta1_gan, theta2_gan, theta3_gan, q)
                curr_batch = np.vstack((batch[-batchsize:, :], images))
                curr_batchY = np.hstack((np.random.randint(0, 100, (images.shape[0])) * 0.001, np.random.randint(900, 1000, (images.shape[0])) * 0.001)).reshape(batchY[-batchsize:].shape[0] * 2, 1)
                #addings instance noise in order to help make the distributions overlapp more and lead to better convergence
                curr_batch += np.random.randn(curr_batch.shape[0], curr_batch.shape[1]) * noiseLevel
                #
                J, accuracy, predValues, z2, z1, a2, a1 = discriminatorForward(curr_batch, curr_batchY, _lambda, theta1, theta2, theta3, q)

            if j % 3 == 0: #%4
                #every second run, we optimize the generator
                theta1_grad, theta2_grad, theta3_grad, delta1 = discriminatorBackwards(curr_batch[currSize :, :], 0 * curr_batchY[currSize :, :], _lambda, theta1, theta2, theta3, predValues[ currSize :, :], z2[ currSize :, :], z1[ currSize :, :], a2[ currSize :, :], a1[ currSize:, :])
                theta1_gan_grad, theta2_gan_grad, theta3_gan_grad = generatorBackwards(theta1, theta3_gan, theta2_gan, theta1_gan, images, delta1, z2_gen, z1_gen, a2_gen, a1_gen, noise, np.maximum(batchsize, 1), _lambda)
                theta1_gan -= alpha*theta1_gan_grad
                theta2_gan -= alpha*theta2_gan_grad
                theta3_gan -= alpha*theta3_gan_grad
               
            else:
                #every first run, we optimize the discriminator
                theta1_grad, theta2_grad, theta3_grad, delta1 = discriminatorBackwards(curr_batch, curr_batchY, _lambda, theta1, theta2, theta3, predValues, z2, z1, a2, a1)
                theta3 -= alpha*theta3_grad
                theta2 -= alpha*theta2_grad
                theta1 -= alpha*theta1_grad

            averageError += J / (len(train_Y)//batchsize + 1)
            averageAccuracy += accuracy / (len(train_Y)//batchsize + 1)
        print(theta1_gan, theta1_gan_grad)
        gan_history = np.vstack((gan_history, images[0]))
        print("At epoch",i,":", averageError)
        print("avg. Accuracy:", np.round(averageAccuracy,3))
    return theta3, theta2, theta1, noise_shape, theta3_gan, theta2_gan, theta1_gan, gan_history

def main():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
    train_X = (train_X - np.min(train_X)) / (np.max(train_X) - np.min(train_X))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
    test_X = (test_X - np.min(test_X)) / (np.max(test_X) - np.min(test_X))
    #get only the 7 - columns, as that is the number we want to generate
    not_0 = np.where(train_y != 0)
    train_X = np.delete(train_X, not_0, axis = 0)
    not_0 = np.where(test_y != 0)
    test_X = np.delete(test_X, not_0, axis = 0)
    test_y, train_y = np.zeros((test_X.shape[0], )), np.zeros((train_X.shape[0], ))
    
    _lambda = 0.01 #0.01
    iterations = 220
    alpha = 0.0001 #0.0001
    batchsize = 1 #1
    
    theta3, theta2, theta1, noise_shape, theta3_gan, theta2_gan, theta1_gan, gan_history = nn_runner(train_X, train_y, iterations, _lambda, alpha, alpha_gen, batchsize)
if __name__ == '__main__':
    main()
