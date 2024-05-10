import piquasso as pq

import numba as nb
import numpy as np

import matplotlib.pyplot as plt

from piquasso.cvqnn import (
    generate_random_cvqnn_weights,
    create_layers,
)
from datetime import datetime
import os
import json
import time

max_d = 10
layer_count = 1
seed_sequence = 1234

shots = 100


now = datetime.now()

folder_name = "data"

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

FILENAME = f"{folder_name}/benchmark_{now.day}_{now.hour}_{now.minute}_{max_d}_{layer_count}_{shots}_{seed_sequence}.json"


def get_samples(simulator, weights, shots):
    layers = create_layers(weights)

    with pq.Program() as program:
        pq.Q() | pq.Vacuum()

        pq.Q() | layers

        pq.Q(*tuple(range(simulator.d))) | pq.HomodyneMeasurement()

    return simulator.execute(program, shots).samples


def generate_cvqnn_samples(simulator, d):
    weights_to_learn = generate_random_cvqnn_weights(
        layer_count, d, active_var=1.0, passive_var=np.pi, rng=simulator.config.rng
    )

    samples = get_samples(simulator, weights_to_learn, shots)

    return samples


def save_data(data):
    with open(FILENAME, "w") as f:
        json.dump(data, f)


def benchmark():
    ITER = 100
    warmup = 10

    times = {"means": [], "range": []}

    for d in range(1, 11):
        print("d:", d)
        runtimes = []
        cutoff = 7

        if d == 7:
            ITER = 1
            warmup = 1

        for i in range(ITER + warmup):
            simulator = pq.PureFockSimulator(
                d=d,
                config=pq.Config(
                    cutoff=cutoff, seed_sequence=seed_sequence + cutoff + i
                ),
            )

            start_time = time.time()
            generate_cvqnn_samples(simulator, d)
            if i >= warmup:
                runtimes.append(time.time() - start_time)


        mean_runtime = np.mean(runtimes)
        print(mean_runtime)

        times["means"].append(mean_runtime)
        times["range"].append(d)

        save_data(times)


if __name__ == "__main__":
    benchmark()
