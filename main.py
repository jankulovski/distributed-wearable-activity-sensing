import csv
import socket
import select
import numpy as np
import matplotlib.pyplot as plt

from construct import Array
from construct import Float32n

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neural_network import MLPClassifier
import operator

from collections import defaultdict


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
        self.zero_cross_rate = sum([(self.payload[j] - self.mean) * (self.payload[j-1] - self.mean) for j, _ in
                                    enumerate(self.payload)])

    def data(self):
        # List representation of sample
        return [self.mean, self.std_deviation, self.zero_cross_rate]


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

        self.ml_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

        if not __test__:
            self.ml_model.fit(self.data_set.data, self.data_set.target)

        self.stream_payload = None

    def predict(self):
        return self.ml_model.predict_proba(np.array([self.stream_payload, ]))


# Test mode flag
__test__ = False

# Socket buffer size
socket_buffer_size = 128 * 1024

# Packet structure
packet = Array(10, Float32n)

# Boards configuration
boards = [
    Board(ip='192.168.43.152', port=8888, board_ip='192.168.43.25', data_set='data/board_0.data'),  # 21 Left p
    Board(ip='192.168.43.152', port=8887, board_ip='192.168.43.243', data_set='data/board_1.data'),  # 29 Right p
    Board(ip='192.168.43.152', port=8886, board_ip='192.168.43.82', data_set='data/board_2.data'),  # 28 Ch
]

# boards lookup table
boards_lookup_table = {}

# Activity predictions for each board
predictions = {}

# Activities
activities = {
    0: 'standby',
    1: 'walk',
    2: 'run'
}

# states transitions
states = {
    0: 3,
    1: 4,
    2: 5,
    3: {1: 4, 2: 5, 0: 0},
    4: {1: 1, 2: 5, 0: 3},
    5: {1: 4, 2: 2, 0: 3},
}

# keep track of the current state
current_state = 0

# Plot settings
plot_means = (0, 0, 0)

plot_ind = np.arange(len(plot_means))
plot_width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(plot_ind - plot_width/2, plot_means, plot_width, color='SkyBlue', label='Standby')
rects2 = ax.bar(plot_ind + plot_width/2, plot_means, plot_width, color='IndianRed', label='Walk')
rects3 = ax.bar(plot_ind + plot_width+(plot_width/2), plot_means, plot_width, color='Green', label='Run')

ax.set_xticks(plot_ind+0.5*plot_width)
ax.set_xticklabels(('board #21', 'board #29', 'board #28'), fontdict=None, minor=False)
ax.set_ylim([-2, 2])

ax.set_ylabel('Activity Probabilities')
ax.set_xlabel('Boards')
ax.set_title('Sensing')
ax.legend()

rects = [rects1, rects2, rects3]


# vote for the final decision
def vote_final_decision(_predictions):

    votes = defaultdict(int)

    for board_predictions in _predictions:
        for prediction in board_predictions:
            votes[board_predictions[prediction]['predicted_activity']] += 1

    max_votes = max(votes.values())
    vts = [(key, value) for key, value in votes.items() if value == max_votes]
    if len(vts) == 1:
        return vts[0][0]
    else:
        max_prob = 0
        activity_with_max_prob = None
        for vote in vts:
            for board_predictions in _predictions:
                for prediction in board_predictions:
                    if board_predictions[prediction]['predicted_activity'] == vote[0]:
                        if board_predictions[prediction]['prediction_probability'] > max_prob:
                            max_prob = board_predictions[prediction]['prediction_probability']
                            activity_with_max_prob = board_predictions[prediction]['predicted_activity']

        return activity_with_max_prob


# Update plots
def update_plots(_predictions):

    for board_predictions in _predictions:
        for board_index in board_predictions:

            index = 0

            for rect in rects:
                if board_predictions[board_index]['probabilities'].shape[0] <= index:
                    height = 0
                else:
                    height = float(board_predictions[board_index]['probabilities'][index])

                rect[board_index].set_height(height)
                index += 1

    fig.canvas.draw()
    plt.pause(0.01)


def run():

    global current_state

    # List of temporary boards predictions
    boards_predictions = []

    reciv_addrs = set()

    while True:

        ready_socks, _, _ = select.select([board.sock for board in boards], [], [])

        for _sock in ready_socks:

            # Read stream packets from socket (non-blocking)
            stream_data, address = _sock.recvfrom(socket_buffer_size)

            if address in reciv_addrs:
                continue

            reciv_addrs.add(address)

            # Parse and collect streamed packets
            boards[boards_lookup_table[address[0]]].stream_payload = Sample(stream_data).data()

            if __test__:
                # Store test/train data
                with open(boards[boards_lookup_table[address[0]]].data_set.fname, 'a') as f:
                    a = boards[boards_lookup_table[address[0]]].stream_payload
                    f.write(",".join(map(str, a)) + ", 2\n")
            else:
                # Perform model prediction
                board_predictions = boards[boards_lookup_table[address[0]]].predict()

                # Find the activity with max probability
                activity_index, activity_probability = max(enumerate(board_predictions[0]), key=operator.itemgetter(1))

                boards_predictions.append({boards_lookup_table[address[0]]: {
                    'probabilities': board_predictions[0],
                    'predicted_activity': activity_index,
                    'prediction_probability': activity_probability
                }})

        if not __test__:

            if len(reciv_addrs) == len(boards):

                # vote for final decision
                final_decision_activity = vote_final_decision(boards_predictions)

                if final_decision_activity != current_state:

                    if type(states[current_state]) == dict:
                        current_state = states[current_state][final_decision_activity]
                    else:
                        current_state = states[final_decision_activity]

                if current_state == 0:
                    txt = "Stationary"
                elif current_state == 3:
                    txt = "Weak Stationary"
                elif current_state == 1:
                    txt = "Walk"
                elif current_state == 4:
                    txt = "Weak Walk"
                elif current_state == 2:
                    txt = "Run"
                elif current_state == 5:
                    txt = "Weak Run"
                else:
                    txt = "Unknown"

                print("Current state: ", txt)

                update_plots(boards_predictions)

                # reset predictions list
                boards_predictions = []
                reciv_addrs.clear()


if __name__ == '__main__':
    boards_lookup_table = {board.board_ip: index for index, board in enumerate(boards)}
    run()
