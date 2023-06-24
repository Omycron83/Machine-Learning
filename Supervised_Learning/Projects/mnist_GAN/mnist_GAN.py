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

@njit(parallel=True)
def generatorForward(noise, theta1_gan, theta2_gan, theta3_gan, q):
    #on every layer (except output), we drop-out 1-q % of values of neurons (and  1 - q - 0.3 in the input layer as we shouldnt drop out too many features)
    if q != 0:
        train_u = np.random.rand(np.shape(noise)[0], np.shape(noise)[1])
        noise *= (train_u > q) / (1 - q)
    train_X = np.hstack((np.ones((np.shape(noise)[0],1)), noise))
    #forward propagation
    z1 = con(train_X)@con(theta1_gan.T)
    if q != 0:
        u1 = np.random.rand(np.shape(z1)[0], np.shape(z1)[1])
        z1 *= (u1 > q) / (1 - q)
    a1 = np.hstack((np.ones((np.shape(z1)[0],1)), ReLU(z1)))
    z2 = con(a1)@con(theta2_gan.T)
    if q != 0:
        u2 = np.random.rand(np.shape(z2)[0], np.shape(z2)[1])
        z2 *= (u2 > q) / (1 - q)
    a2 = np.hstack((np.ones((np.shape(z2)[0],1)), ReLU(z2)))
    images = sigmoid(con(a2)@con(theta3_gan.T))
    return z1, z2, a1, a2, images

@njit(parallel=True)
def generatorBackwards(theta1, theta3_gan, theta2_gan, theta1_gan, images, prevLayer, z2, z1, a2, a1, noise, batchsize, _lambda):
    noise = np.hstack((np.ones((np.shape(noise)[0],1)), noise))
    delta3 = con(prevLayer) @ con(theta1[:,1:]) *  sigmoidDeriv(images)
    delta2 = con(delta3) @ con(theta3_gan[:,1:]) * ReLUDeriv(z2)
    delta1 = con(delta2) @ con(theta2_gan[:,1:]) * ReLUDeriv(z1)
    D3 = con(delta3.T) @ con(a2)
    D2 = con(delta2.T) @ con(a1)
    D1 = con(delta1.T) @ con(noise)
    theta3_gan_grad = D3 / batchsize + np.hstack((np.zeros((np.shape(theta3_gan)[0], 1)), (_lambda/batchsize)*np.absolute(theta3_gan[:,1:])))
    theta2_gan_grad = D2 / batchsize + np.hstack((np.zeros((np.shape(theta2_gan)[0], 1)), (_lambda/batchsize)*np.absolute(theta2_gan[:,1:])))
    theta1_gan_grad = D1 / batchsize + np.hstack((np.zeros((np.shape(theta1_gan)[0], 1)), (_lambda/batchsize)*np.absolute(theta1_gan[:,1:])))
    return theta1_gan_grad, theta2_gan_grad, theta3_gan_grad

@njit(parallel=True)
def discriminatorBackwards(train_X, train_Y, _lambda, theta1, theta2, theta3, predValues, z2, z1, a2, a1):
    train_X = np.hstack((np.ones((np.shape(train_X)[0],1)), train_X))
    delta3 = predValues - train_Y 
    delta2 = con(delta3) @ con(theta3[:,1:]) * ReLUDeriv(z2)
    delta1 = con(delta2) @ con(theta2[:,1:]) * ReLUDeriv(z1)
    D3 = con(delta3.T) @ con(a2)
    D2 = con(np.transpose(delta2)) @ con(a1)
    D1 = con(np.transpose(delta1)) @ con(train_X)
    #now we take the average gradient, multiply by the inverse of the probability of that node not being picked in order to account for some heaving been more relevant than others and then add regularization
    theta3_grad = D3 / len(train_Y) + np.hstack((np.zeros((np.shape(theta3)[0], 1)), (_lambda/len(train_Y))*np.absolute(theta3[:,1:])))
    theta2_grad = D2 / len(train_Y) + np.hstack((np.zeros((np.shape(theta2)[0], 1)), (_lambda/len(train_Y))*np.absolute(theta2[:,1:])))
    theta1_grad = D1 / len(train_Y) + np.hstack((np.zeros((np.shape(theta1)[0], 1)), (_lambda/len(train_Y))*np.absolute(theta1[:,1:])))
    return theta1_grad, theta2_grad, theta3_grad, delta1

@njit(parallel=True)
def discriminatorForward(train_X, train_Y, _lambda, theta1, theta2, theta3, q):
    #on every layer (except output), we drop-out 1-q % of values of neurons (and  1 - q - 0.3 in the input layer as we shouldnt drop out too many features)
    if q != 0:
        train_u = np.random.rand(np.shape(train_X)[0], np.shape(train_X)[1])
        train_X *= (train_u > (np.maximum(q - 0.3, 0))) / (1 - np.maximum(q - 0.3, 0))
    train_X = np.hstack((np.ones((np.shape(train_X)[0],1)), train_X))
    #forward propagation
    z1 = con(train_X)@con(np.transpose(theta1))
    if q != 0:
        u1 = np.random.rand(np.shape(z1)[0], np.shape(z1)[1])
        z1 *= (u1 > q) / (1 - q)
    a1 = np.hstack((np.ones((np.shape(z1)[0],1)), ReLU(z1)))

    z2 = con(a1)@con(np.transpose(theta2))
    if q != 0:
        u2 = np.random.rand(np.shape(z2)[0], np.shape(z2)[1])
        z2 *= (u2 > q) / (1 - q)
    a2 = np.hstack((np.ones((np.shape(z2)[0],1)), ReLU(z2)))
    #last step
    predValues = sigmoid(con(a2)@con(np.transpose(theta3)))
    J = np.sum(-train_Y*np.log(predValues)  -  (1-train_Y)*np.log(1 - predValues)) / len(train_Y)
    #in order to calculate accuracy, we just assume that the rounded value of our predicted values dictates if its value is 1 or 0 and look if that's the same as in Y
    accuracy = np.sum(rounds(predValues[:predValues.shape[0] // 2]) == rounds(train_Y[:predValues.shape[0] // 2])) / len(train_Y[:predValues.shape[0] // 2])
    #calculating the cost with regularization
    J += np.sum(_lambda/(2*len(train_Y)) * theta1[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta2[:,1:]**2)
    J += np.sum(_lambda/(2*len(train_Y)) * theta3[1:]**2)
    return J, accuracy, predValues, z2, z1, a2, a1
 
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
    alpha_gen = 0.001 #0.001
    batchsize = 1 #1
    
    theta3, theta2, theta1, noise_shape, theta3_gan, theta2_gan, theta1_gan, gan_history = nn_runner(train_X, train_y, iterations, _lambda, alpha, alpha_gen, batchsize)
    #
    noise = np.random.randn(test_X.shape[0], noise_shape)
    z1_gen, z2_gen, a1_gen, a2_gen, images = generatorForward(noise, theta1_gan, theta2_gan, theta3_gan, 0)
    test_X = np.vstack((test_X, images))
    test_y = np.hstack((test_y, np.ones(images.shape[0]))).reshape(test_X.shape[0], 1)
    #
    J, accuracy, predValues, z2, z1, a2, a1 = discriminatorForward(test_X, test_y, _lambda, theta1, theta2, theta3, 0)
    print("And an accuracy on test of", accuracy, "with an error of", J)
    fig = plt.figure()
    rows = int(gan_history.shape[0] ** 0.5) + 1
    columns = int(gan_history.shape[0] ** 0.5) + 1
    for i in range(gan_history.shape[0]):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(gan_history[i].reshape(28, 28), cmap = 'gray')
        plt.axis('off')
    plt.show()

    for i in range(images.shape[0]):
        plt.imshow(images[i].reshape(28, 28), cmap = 'gray')
        plt.show()
    
        
    print(theta1, theta2, theta3, theta1_gan, theta2_gan, theta3_gan)
    np.save("theta1", theta1)
    np.save("theta2", theta2)
    np.save("theta3", theta3)
    np.save("theta1_gan", theta1_gan)
    np.save("theta2_gan", theta2_gan)
    np.save("theta3_gan", theta3_gan)
if __name__ == '__main__':
    main()
