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

import numpy as np

import piquasso as pq

import matplotlib as mpl
import matplotlib.pyplot as plt

from d_2_script import get_samples

from utils import gaussian_mmd_grad


plt.rcParams.update({"font.size": 13})
mpl.rc("text", usetex=True)

d = 2
seed_sequence = 12345
cutoff = 10


if __name__ == "__main__":
    simulator = pq.PureFockSimulator(
        d=d, config=pq.Config(cutoff=cutoff, seed_sequence=seed_sequence)
    )

    weights_to_learn = np.array([1.0, 0.1, np.pi / 6])

    weights = np.zeros_like(weights_to_learn)
    initial_weights = np.copy(weights)

    s_D = 1.0
    s_S = 1.0

    shift_amounts = [s_D, s_S, np.pi / 2]
    multipliers = [1.0 / s_D, 1.0 / np.sinh(s_S), 1.0]

    shot_list = np.arange(10, 2000, 10).tolist()

    gradient_array = np.zeros(shape=(len(shift_amounts), len(shot_list)))

    for j in range(len(shift_amounts)):
        print(f"Executing with weight no. {j+1}")

        for k, sample_shots in enumerate(shot_list):
            samples_to_learn = get_samples(simulator, weights_to_learn, sample_shots)
            samples = get_samples(simulator, weights, sample_shots)

            weights[j] += shift_amounts[j]

            plus_samples = get_samples(simulator, weights, sample_shots)

            weights[j] -= 2 * shift_amounts[j]

            minus_samples = get_samples(simulator, weights, sample_shots)

            weights[j] += shift_amounts[j]

            mmd_grad = gaussian_mmd_grad(
                plus_samples, minus_samples, samples, samples_to_learn
            )

            gradient = multipliers[j] * mmd_grad

            gradient_array[j, k] = gradient


    fig, axis = plt.subplots(nrows=len(shift_amounts), sharex=True)

    for j in range(len(shift_amounts)):
        axis[j].plot(shot_list, gradient_array[j])

        axis[j].set_ylabel(f"$w_{j+1}$")

    plt.xlabel("Shots")

    plt.show()
