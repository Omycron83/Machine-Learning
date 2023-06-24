import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualization(b, m, datax, datay):
    x = np.linspace(min(datax), max(datax), 50)
    y = m*x + b
    plt.plot(x,y, '-r')
    plt.plot(datax, datay, 'o', color = 'blue')
    plt.title("Line of best fit für gewicht/höhe")
    plt.xlabel("Gewicht in Pfund")
    plt.ylabel("Höhe in Zoll")
    plt.show()
                 

def gradientDescent(X, y, theta, learningRate):
    for i in range(3000):
        theta -= learningRate/len(y) * np.sum((np.stack((X @ theta - y, X @ theta - y), axis = -1) * X), axis = 0)
    return theta



def main():
    data1 = pd.read_csv("weight-height.csv")
    datay = data1['Height'].tolist()
    datax = data1['Weight'].tolist()
    x_bias = np.ones((len(datax),1))

    x_new = np.reshape(datax,(len(datax),1))
    x_new = np.append(x_bias, x_new, axis = 1)

    x_new_transpose = np.transpose(x_new)
    x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)

    temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)

    temp_2 = x_new_transpose.dot(datay)
    theta = temp_1.dot(temp_2)

    theta2 = gradientDescent(x_new,np.array(datay), [0, 0], 0.000001)
    intercept2 = float(theta2[0])
    slope2 = float(theta2[1])
    print(intercept2)
    print(slope2)
    
    
    intercept = theta[0]
    slope = theta[1]
    print("Intercept:", intercept)
    print("Slope:", slope)
    visualization(intercept, slope, datax, datay)
    visualization(intercept2, slope2, datax, datay)
    


if __name__ == '__main__':
	main()
