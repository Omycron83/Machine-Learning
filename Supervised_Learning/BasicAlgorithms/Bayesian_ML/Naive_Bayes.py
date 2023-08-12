
class NaiveBayes_Class_Bin:
    def __init__(self, data):
        #Data: [x1, x2, ..., xn, Class]
        self.data = data
    def add_data(self, data):
        self.data += data
    def construct_model(self):
        #Using bayes rule: P(Class | x) = (P(x | Class) * P(Class)) / P(x)
        #Applying this multivariatly, i.e. P(Class | x1, x2, ..., xn):
        #We can use the chain rule of conditional probability: P(x1 | x2, ..., xn, Class) * P(x2, ..., xn, Class) = P(x1 | x2, ...)*P(x2 | x3...) * P(x3 ...)
        #P(Class | x1, ..., xn) is proportional to P(x1 | Class) * P(x2 | class)...* P(Class)
        #So all in all, we need: amount_class1 / m, amount_class2 / m...
        #As well as amount_x1 in class1 / amount_class1 ...
        #We can store amount_class1 etc. in class_dim list, where the index in the list is determined by the feature variable
        #Then, we need to store the amount of features in each class, determined by an classes x feature variables 
        #We will fill this with 1s first though, and only after the first pass to determine the amount of classes

        self.classes = []
        for i in range(len(self.data)):
            if self.data[i][-1] <= len(self.classes) - 1:
                self.classes[self.data[i][-1]] += 1
            else:
                for j in range(self.data[i][-1] - len(self.classes) + 1):
                    self.classes.append(0)
                self.classes[self.data[i][-1]] += 1
        #Constructing the feature counts
        self.features = []
        for i in range(len(self.classes)):
            self.features.append([1 for J in range(len(self.data[0]) - 1)])
        #Inserting them
        for i in range(len(self.data)):
            for j in range(len(self.data[0]) - 1):
                self.features[self.data[i][-1]][j] += self.data[i][j]
        
        #"Normalizing":

        print(self.classes, self.features)

data = [[1, 0, 3],[4, 2, 1],[6, 5, 2],[2, 4, 3]]

x = NaiveBayes_Class_Bin(data)
x.construct_model()