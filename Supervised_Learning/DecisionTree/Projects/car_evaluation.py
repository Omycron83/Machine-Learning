import pandas as pd
import DecisionTree as DT
data = pd.read_csv("D:\Damian\PC\Python\ML\DecisionTree\car_evaluation.csv")
buying_price = {"vhigh": 3, "high" : 2, 'med' : 1,'low':0}
maintenance_cost = {"vhigh": 3, "high" : 2, 'med' : 1,'low':0}
num_doors = {"4": 4, "3" : 3, '2' : 2,'5more': 5}
num_pers = {"2": 2, "4" : 4, 'more' : 5}
lug_boot = {"big": 2, 'med' : 1,'small':0}
safety = {"high" : 2, 'med' : 1,'low':0}
decision = {"vgood": 1, "good" : 1, 'acc' : 1,'unacc':0}

buying_price_list = data["vhigh"].values.tolist()
maintenance_cost_list = data["vhigh.1"].values.tolist()
num_doors_list = data["2"].values.tolist()
num_pers_list = data["2.1"].values.tolist()
lug_boot_list = data["small"].values.tolist()
safety_list = data["low"].values.tolist()
decision_list = data["unacc"].values.tolist()

data = []
for i in range(len(buying_price_list)):
    data.append([i, buying_price[buying_price_list[i]], 
    maintenance_cost[maintenance_cost_list[i]], num_doors[num_doors_list[i]], num_pers[num_pers_list[i]],
    lug_boot[lug_boot_list[i]],safety[safety_list[i]], decision[decision_list[i]]]) 

training, test = DT.get_split(data)

tree = DT.classification_tree(training, 100000)
print(tree.get_accuracy(test))

forest = DT.classification_forest(training, 1)
print(forest.get_accuracy(test))