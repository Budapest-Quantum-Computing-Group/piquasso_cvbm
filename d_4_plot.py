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

import sys
import json

import numpy as np

import piquasso as pq

from d_4_script import get_samples

from utils import gaussian_mmd

import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 13})
mpl.rc("text", usetex=True)


def _estimate_loss_with_itself(simulator, weights, shots, N):
    sum_ = 0.0

    samples = get_samples(simulator, weights, 2 * N * shots)

    for i in range(N):
        sum_ += gaussian_mmd(
            samples1=samples[2 * i * shots : (2 * i + 1) * shots],
            samples2=samples[(2 * i + 1) * shots : (2 * i + 2) * shots],
        )

    return sum_ / N


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        data = json.load(f)

    simulator = pq.PureFockSimulator(
        d=data["d"], config=pq.Config(cutoff=data["cutoff"])
    )

    ITER = len(data["losses"])

    losses = data["losses"]

    weight_history = np.array(data["weight_history"])

    x = np.arange(weight_history.shape[0])

    weights_to_learn = data["weights_to_learn"]

    loss_with_itself = _estimate_loss_with_itself(
        simulator, weights_to_learn, data["shots"], N=10
    )

    fig, (axup, axdown) = plt.subplots(nrows=2, sharex=True)

    loss_with_itself_arr = np.full_like(losses, fill_value=loss_with_itself)

    axup.scatter(x, losses, s=5, marker=".")
    axup.plot(x, loss_with_itself_arr, "r")
    axup.set_xlabel("Number of iterations")
    axup.set_ylabel("Loss")

    axup.set_xlim((-1, ITER + 1))
    axup.set_ylim((0, max(losses) + 0.001))

    colors = []

    for idx in range(weight_history.shape[1]):
        p = axdown.plot(x, weight_history[:, idx], label=f"$w_{idx+1}$")
        colors.append(p[0].get_color())

    axdown.legend()

    ax2 = axdown.twinx()
    ax2.set_yticks(
        weights_to_learn,
        [f"$w_{idx+1}^*$" for idx in range(weight_history.shape[1])],
    )

    for ytick, color in zip(ax2.get_yticklabels(), colors):
        ytick.set_color(color)

    ylim = (
        min(np.min(weights_to_learn), np.min(weight_history)) - 0.5,
        max(np.max(weights_to_learn), np.max(weight_history)) + 0.5,
    )

    ax2.set_ylim(ylim)

    ax2.set_ylabel("Weights of training data")

    axdown.set_xlabel("Number of iterations")
    axdown.set_ylabel("Weights")
    axdown.set_xlim((0, ITER))
    axdown.set_ylim(ylim)

    fig.show()
    plt.show()
