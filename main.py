import csv
import socket
import select
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


# Socket buffer size
socket_buffer_size = 128 * 1024

# List of active socks
socks = []

# Packet structure
packet = Array(10, Float32n)

# Boards configuration
boards = [
    {'id': 0, 'ip': '172.20.10.9', 'port': 8887},
    {'id': 1, 'ip': '172.20.10.9', 'port': 8888},
    {'id': 2, 'ip': '172.20.10.9', 'port': 8889},
]

# Boards lookup table
boards_lookup_table = {
    '172.20.10.9:8887': 0,
    '172.20.10.9:8888': 1,
    '172.20.10.9:8889': 2,
}

# Boards buckets
boards_buckets = {}

# ML Model
ml_model = None


# Retrieves socket packets - non blocking
def packets(_sock):
    while True:
        # noinspection PyBroadException
        try:
            yield _sock.recvfrom(socket_buffer_size)
        except Exception:
            return


# Initialize sockets for all boards
def init_socks():
    for board in boards:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _sock.bind((board['ip'], board['port']))
        _sock.setblocking(0)
        socks.append(_sock)


# UDP Listen to all boards
def listen():
    while True:
        for _sock, _, _ in select.select(socks, [], []):
            for data, address in packets(_sock):
                boards_buckets[address].append(packet.parse(data))
                # Make sure to clean boards buckets as of fast growth


# Init boards buckets
def init_buckets():
    for board in boards_lookup_table.keys():
        boards_buckets[board] = []


# Data set structure
class DataSet:
    def __init__(self, fname):
        self.data = None
        self.target = None
        self.load_data(fname)

    def load_data(self, fname):
        # Loads (csv) data from a given filename. Last axis is label.
        data, target = [], []
        for row in csv.reader(open(fname), delimiter=','):
            data.append(row[:-1])
            target.append(row[-1])

        self.data = np.array(data).astype(float)
        self.target = np.array(target).astype(int)

    def min(self, axis):
        # Returns min value for a given axis
        return self.data[:, axis].min()

    def max(self, axis):
        # Returns max value for a given axis
        return self.data[:, axis].max()

    def mean(self, axis):
        # Returns max value for a given axis
        return self.data[:, axis].mean()


def init_ml_model(data):
    global ml_model
    ml_model = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=3, n_init=10,
                               max_no_improvement=10, verbose=0, random_state=0).fit(data.data, data.target)


if __name__ == '__main__':
    # init_ml_model(DataSet("data/v1.data"))
    # init_socks()
    # init_buckets()
    # listen()
    pass

###########################################################################################

X = DataSet("data/v1.data")

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
