#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time

from datetime import datetime

import piquasso as pq
from piquasso.cvqnn import get_cvqnn_weight_indices

import numpy as np

from utils import AdamOptimizer, gaussian_mmd_grad, gaussian_mmd


now = datetime.now()

folder_name = "data"

d = 2
shots = 1000
diff_shots = shots
learning_rate = 0.005
seed_sequence = 12345
cutoff = 10

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

FILENAME = f"{folder_name}/2_mode_{now.day}_{now.hour}_{now.minute}_{shots}_{seed_sequence}.json"


def save_data(data):
    with open(FILENAME, "w") as f:
        json.dump(data, f)


def get_samples(simulator, weights, shots):
    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q(0) | pq.CubicPhase(6.0)
        pq.Q(1) | pq.CubicPhase(3.0)

        pq.Q(0, 1) | pq.CrossKerr(np.pi / 6)

        pq.Q(0) | pq.Displacement(weights[0])
        pq.Q(1) | pq.Squeezing(weights[1])

        pq.Q(0, 1) | pq.Beamsplitter(weights[2])

        pq.Q() | pq.HomodyneMeasurement()

    samples = simulator.execute(program, shots).samples

    return samples


if __name__ == "__main__":
    ITER = 600

    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff, seed_sequence=seed_sequence)
    )

    weights_to_learn = np.array([1.0, 0.1, np.pi / 6])

    weights = np.zeros_like(weights_to_learn)
    initial_weights = np.copy(weights)

    weight_indices = get_cvqnn_weight_indices(d)

    s_D = 1.0
    s_S = 1.0

    shift_amounts = [s_D, s_S, np.pi / 2]
    multipliers = [1.0 / s_D, 1.0 / np.sinh(s_S), 1.0]

    optimizer = AdamOptimizer(learning_rate, number_of_weights=len(weights))

    grad_vector = np.empty_like(weights)

    data = {
        "losses": [],
        "weights": [],
        "shots": shots,
        "cutoff": cutoff,
        "learning_rate": learning_rate,
        "ts": s_S,
        "td": s_D,
        "d": d,
        "weights_to_learn": weights_to_learn.tolist(),
        "seed_sequence": seed_sequence,
        "weight_history": [],
        "time": None,
    }

    start_time = time.time()
    samples = get_samples(simulator, weights, shots)

    for i in range(ITER):
        samples_to_learn = get_samples(simulator, weights_to_learn, shots)
        for j in range(len(weights)):
            weights[j] += shift_amounts[j]

            plus_samples = get_samples(simulator, weights, diff_shots)

            weights[j] -= 2 * shift_amounts[j]

            minus_samples = get_samples(simulator, weights, diff_shots)

            weights[j] += shift_amounts[j]

            mmd_grad = gaussian_mmd_grad(
                plus_samples, minus_samples, samples, samples_to_learn
            )

            grad_vector[j] = multipliers[j] * mmd_grad

        weights = optimizer.update_weights(weights, grad_vector)

        samples = get_samples(simulator, weights, shots)

        if True:
            loss = gaussian_mmd(samples_to_learn, samples)

            print("-----------------")
            print(f"{i}. loss: {loss}")
            print("weights:", weights)
            print("grad:", grad_vector)

            data["losses"].append(loss)
            data["weights"] = [weights.tolist()]
            data["weight_history"].append(weights.tolist())
            data["time"] = time.time() - start_time

            save_data(data)
