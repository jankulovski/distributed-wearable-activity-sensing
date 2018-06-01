import csv
import socket
import select
import operator
import numpy as np
import matplotlib.pyplot as plt

from pylab import figure

from construct import Array
from construct import Float32n

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score

from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import train_test_split


# Data set structure
class DataSet:
    def __init__(self, fname):
        self.data = None
        self.target = None
        self.fname = fname
        self.load_data(fname)

    def load_data(self, fname):
        # Loads (csv) data from a given filename. Last axis is label.
        data, target = [], []
        for row in csv.reader(open(fname), delimiter=','):
            data.append(row[:-1])
            target.append(row[-1])

        self.data = np.array(data).astype(float)
        self.target = np.array(target).astype(int)


# Sample structure
class Sample:
    def __init__(self, payload):
        self.payload = np.array(packet.parse(payload))
        self.mean = self.payload.mean()
        self.std_deviation = self.payload.std()
        self.zero_cross_rate = self.zcr()

    def data(self):
        # List representation of sample
        return [self.mean, self.std_deviation, self.zero_cross_rate]

    def zcr(self):
        # Zero-crossing rate
        return sum([(self.payload[j] - self.mean) * (self.payload[j-1] - self.mean) for j, _ in enumerate(self.payload)])


# Board repr.
class Board:
    def __init__(self, ip, port, board_ip, data_set):

        self.ip = ip
        self.port = port
        self.board_ip = board_ip

        self.data_set = DataSet(data_set)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.ip, self.port))

        self.ml_model = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=3, n_init=10,
                                        max_no_improvement=10, verbose=0, random_state=0)
        self.ml_model.fit(self.data_set.data, self.data_set.target)

        self.stream_payload = None

    def predict(self):
        return self.ml_model.predict(np.array([self.stream_payload, ]))


# Test mode flag
__test__ = False

# Socket buffer size
socket_buffer_size = 128 * 1024

# Packet structure
packet = Array(10, Float32n)

# Boards configuration
boards = [
    Board(ip='192.168.0.103', port=8888, board_ip='192.168.0.108', data_set='data/board_0.data'),
    # Board(ip='192.168.0.103', port=8888, board_ip='192.168.0.109', data_set='data/board_0.data'),
    # Board(ip='192.168.0.103', port=8888, board_ip='192.168.0.110', data_set='data/board_0.data'),
]

# boards lookup table
boards_lookup_table = {}

# Activity predictions for each board
predictions = {}

# Activities
activities = {
    0: 'standby',
    1: 'walk',
    3: 'run'
}


def run():
    while True:
        ready_socks, _, _ = select.select([board.sock for board in boards], [], [])
        for _sock in ready_socks:
            stream_data, address = _sock.recvfrom(socket_buffer_size)
            boards[boards_lookup_table[address]].stream_payload = Sample(stream_data).data()

            if __test__:
                with open(boards[boards_lookup_table[address]].data_set.fname, 'a') as f:
                    f.write(boards[boards_lookup_table[address]].stream_payload)

            boards[boards_lookup_table[address]].predict()


if __name__ == '__main__':
    boards_lookup_table = {(board.board_ip, board.port): index for index, board in enumerate(boards)}
    run()


###########################################################################################

X = DataSet("data/board_0.data")

"""
n_clusters : int, optional, default: 8
    The number of clusters to form as well as the number of
    centroids to generate.

init : {'k-means++', 'random' or an ndarray}
    Method for initialization, defaults to 'k-means++':

    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.

    'random': choose k observations (rows) at random from data for
    the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features)
    and gives the initial centers.

n_init : int, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.
    
random_state : int, RandomState instance or None, optional, default: None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.
"""

# model = KMeans(n_clusters=3, random_state=1, n_init=1, init='random').fit(X.data, X.target)
model = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=3,
                        n_init=10, max_no_improvement=10, verbose=0,
                        random_state=0).fit(X.data, X.target)

Y = model.predict(X.data)

print(confusion_matrix(X.target, Y, labels=[0, 1, 2]))
print(accuracy_score(X.target, Y))

# homogeneity: each cluster contains only members of a single class.
print(homogeneity_score(X.target, Y))

# completeness: all members of a given class are assigned to the same cluster.
print(completeness_score(X.target, Y))

# cluster's centers
# print(kmeans.cluster_centers_)

# X_train, X_test, y_train, y_test = train_test_split(X.data, X.target, test_size=0.1, random_state=42)
# print(X_train, X_test, y_train, y_test)

color = ["g", "r", "b"]

fig = figure()
ax = fig.gca(projection='3d')

for i in range(X.data.shape[0]):
    ax.scatter(X.data[i][0], X.data[i][1], X.data[i][2], c=color[model.labels_[i]])

ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2],
           marker="x", s=150, linewidths=5, zorder=100, c="black")
plt.show()
