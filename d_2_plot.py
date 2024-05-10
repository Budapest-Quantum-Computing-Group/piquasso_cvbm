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
import random

import numpy as np

import piquasso as pq

from d_2_script import get_samples

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

    random.seed(data["seed_sequence"])

    simulator = pq.PureFockSimulator(
        d=data["d"], config=pq.Config(cutoff=data["cutoff"])
    )

    ITER = len(data["losses"])

    losses = data["losses"]

    weight_history = np.array(data["weight_history"])

    x = np.arange(weight_history.shape[0])

    weights_to_learn = data["weights_to_learn"]

    loss_with_itself = _estimate_loss_with_itself(
        simulator, weights_to_learn, data["shots"], N=100
    )

    if True:
        fig, ax = plt.subplots()

        loss_with_itself_arr = np.full_like(losses, fill_value=loss_with_itself)

        plt.scatter(x, losses, s=5, marker=".")
        plt.plot(x, loss_with_itself_arr, "r")
        plt.xlabel("Number of iterations")
        plt.ylabel("Loss")

        plt.xlim((-1, ITER + 1))
        plt.ylim((0, max(losses) + 0.01))

        axins = ax.inset_axes(
            [0.4, 0.4, 0.57, 0.57],
            xlim=(200, ITER),
            ylim=(0.000, 0.014),
        )
        axins.scatter(x, losses, s=10, marker=".")
        axins.plot(x, loss_with_itself_arr, "r")
        ax.indicate_inset_zoom(axins, edgecolor="black")

        plt.show()

    if True:
        fig, ax = plt.subplots()

        colors = []

        for idx in range(weight_history.shape[1]):
            p = ax.plot(x, weight_history[:, idx], label=f"$w_{idx+1}$")
            colors.append(p[0].get_color())

        ax.legend()

        ax2 = ax.twinx()
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

        ax.set_xlabel("Number of iterations")
        ax.set_ylabel("Weights")
        ax.set_xlim((0, ITER))
        ax.set_ylim(ylim)

        fig.show()
        plt.show()

    if True:
        plot_shots = 100_000

        learnt_samples = get_samples(
            simulator, weight_history[np.argmin(losses)], plot_shots
        )
        original_samples = get_samples(
            simulator, np.array(data["weights_to_learn"]), plot_shots
        )
        starting_samples = get_samples(
            simulator, np.zeros_like(np.array(data["weights_to_learn"])), plot_shots
        )

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

        ax1.hexbin(
            starting_samples[:, 0],
            starting_samples[:, 1],
            gridsize=100,
            extent=(-2.5, 4.5, -2.5, 2.5),
        )

        ax1.set_title("Initial distribution")

        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")

        ax3.hexbin(
            learnt_samples[:, 0],
            learnt_samples[:, 1],
            gridsize=100,
            extent=(-2.5, 4.5, -2.5, 2.5),
        )

        ax3.set_title("Learned distribution")

        ax3.set_xlabel("$x_1$")
        ax3.set_ylabel("$x_2$")

        ax2.hexbin(
            original_samples[:, 0],
            original_samples[:, 1],
            gridsize=100,
            extent=(-2.5, 4.5, -2.5, 2.5),
        )

        ax2.set_title("Target distribution")

        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")

        f.set_figwidth(15)
        f.tight_layout()

        plt.show()
