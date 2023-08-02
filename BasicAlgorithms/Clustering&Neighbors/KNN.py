
class KNN:
    def norm_loc_vect(self, vec1, vec2):
        norm = 0
        for i in range(len(vec1) - 1):
            norm += (vec1[i] - vec2[i])**2
        return norm**0.5
    #Data: [x1, x2, ..., xn, y]
    def __init__(self, data, k):
        self.data = data
        self.k = k
    def get_max_count_classes(self, data):
        return max(data, key=data[0].count)

    def pred(self, data_point):
        differences = []
        for i in range(len(self.data)):
            differences.append([i, self.norm_loc_vect(self.data[i], data_point)])
        differences.sort(key = lambda x : x[0])
        return self.get_max_count_classes(differences[:self.k])
    

class K_MeansClustering:
    class Vector:
        def __init__(self, liste):
            self.content = liste
        def __add__(self, other):
            new_vec = []
            for i in range(len(self.content)):
                new_vec.append(other.content[i] + self.content[i])
        def __sub__(self, other):
            new_vec = []
            for i in range(len(self.content)):
                new_vec.append(other.content[i] - self.content[i])
        def __mul__(self, other):
            new_vec = []
            for i in range(len(self.content)):
                new_vec.append(other.content[i] * self.content[i])
        def __matmul__(self, other):
            val = 0
            for i in range(len(self.content)):
                val += other.content[i] * self.content[i]
        def __neg__(self):
            for i in range(len(self.content)):
                self.content[i] *= -1
        def norm(self):
            val = 0
            for i in range(len(self.content)):
                val += self.content[i]**2
            return val**0.5

    def __init__(self, data, cluster_amt, iterations):
        self.iterations = iterations
        self.data = data
        self.clusters = []
        for i in range(cluster_amt):
            self.clusters.append(self.Vector(data[i]))
        self.group_clusters()

    def get_lowest_distance(self, datapoint):
        index = 0
        lowest_dist = 1000000000000
        for i in range(self.clusters):
            norm = (self.clusters[i]-datapoint).norm
            if norm > lowest_dist:
                lowest_dist = norm
                index = i
        return index 
            

    def group_clusters(self):
        for i in range(self.iterations):
            for j in range(len(self.data)):
                zugehoerigkeit = []
                zugehoerigkeit.append(self.get_lowest_distance(self.data[j]))
            counts = []
            for j in range(self.clusters):
                self.clusters[j] == 0
                counts.append(zugehoerigkeit.count(j))
            for j in range(len(zugehoerigkeit)):
                self.clusters[zugehoerigkeit[j]] += self.data[j] / counts[zugehoerigkeit[j]]
                


