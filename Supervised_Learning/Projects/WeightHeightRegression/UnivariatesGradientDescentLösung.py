#Programmieraufgabe: Verfolständige die Funktionen gradientDescent und Kostenfunktion.
#Die Funktion "gradientDescentRunner" nimmt dabei die zurückgegebenen Werte m, n nach einem mal gradient descent und berechnet mit diesen die Kostenfunktion.
#Anschließend graphed er die Kostenfunktion über die Anzahl an Iterationen und gibt die optimalen Werte heraus.
#Auch zeigt er die gelernte line of best fit zwischen den Datenpunkten und berechnet einen Hypothetischen Wert.

#Bibliotheken die wir nutzen für das Graphen und das hereinladen der Daten:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
#unser Datensatz beinhaltet eine gekürzte Version des SOCR Human Weight/Height Datensatz (http://socr.ucla.edu/docs/resources/SOCR_Data/SOCR_Data_Dinov_020108_HeightsWeights.html).
#Dort wird die Höhe in Zoll und das Gewicht in Pfund von 500 18 Jährigen angegeben.
#Importieren wir diesen zuerst 

daten = pd.read_csv("SOCR-HeightWeight.csv")
daten = daten * [1, 2.54, 0.453592] #umrechnungswerte von Zoll -> Meter und Pfund -> KG
y = daten['Height(Inches)'].tolist()
x = daten['Weight(Pounds)'].tolist()
alphaN = 0.02 #versucht hier verschiedene Werte für die learning Rate! Ist sie zu niedrig, so wird eure Cost sich kaum verändern, ist sie zu hoch wird sie explodieren!
alphaM = 0.0005
alpha = [alphaM, alphaN]

Iterationen = 100


def Kostenfunktion(x, y, m, n):
    J = np.sum((np.column_stack((np.ones((np.shape(np.array(x))[0],1)), np.array(x))) @ np.array([n, m]).T - np.array(y)) ** 2) / (2*len(y))
    return J


def gradientDescent(x, y, m, n, alpha):
    grad = np.sum( np.column_stack(((np.column_stack((np.ones((np.shape(np.array(x))[0],1)), np.array(x))) @ np.array([n, m]).T - np.array(y)),(np.column_stack((np.ones((np.shape(np.array(x))[0],1)), np.array(x))) @ np.array([n, m]).T - np.array(y))))  * np.column_stack((np.ones((np.shape(np.array(x))[0],1)), np.array(x))), axis = 0) / len(y)
    new = np.array([m, n]) - np.array(alpha) * grad 
    return tuple(new)

def gradientDescentRunner(x, y, iterationen, alpha):
    cost = []
    iterations = []
    m, n = 0, 0
    for i in range(iterationen):
        m, n = gradientDescent(x, y, m, n, alpha)
        cost.append(Kostenfunktion(x, y, m, n))
        iterations.append(i)
    print("Die Kostenfunktion hat sich über die Iterationen so verhalten:")
    plt.plot(iterations, cost, '-r')
    plt.title("Kostenfunktion über Iterationen")
    plt.xlabel("Iteration")
    plt.ylabel("Kostenfunktion")
    plt.show()
    return m, n

m, n = gradientDescentRunner(x, y, Iterationen, alpha)

print("Die line of best fit sieht dann ca. so hier aus:")
line_x = np.linspace(min(x), max(x), 10)
line_y = m * line_x + n
plt.plot(line_x, line_y, '-r')
plt.plot(x, y, 'o', color='blue')
plt.title("Höhe über Gewicht")
plt.xlabel("Gewicht")
plt.ylabel("Höhe")
plt.show()
print("Werte:", "n =", n, "m =", m)
print("Die Kost ist dabei", Kostenfunktion(x, y, m, n))
reg = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))
print(reg.coef_[0], reg.intercept_)
print("Die Optimale Kost ist dabei", Kostenfunktion(x, y, reg.coef_[0], reg.intercept_))

