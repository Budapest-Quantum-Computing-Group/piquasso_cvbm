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

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _gaussian_kernel(x, y, sigma=1.0):
    return np.exp(-np.sum(np.abs(x - y) ** 2) / (2 * sigma**2))


@nb.njit(cache=True)
def _get_kernel_expectation_value(samples, M):
    sum_ = 0.0

    for idx in range(M):
        for jdx in range(idx):
            sum_ += 2 * _gaussian_kernel(samples[idx], samples[jdx])

        sum_ += _gaussian_kernel(samples[idx], samples[idx])

    return sum_ / (M * (M - 1))


@nb.njit(cache=True)
def _get_cross_term(samples1, samples2, M, N):
    sum_ = 0.0

    for idx in range(M):
        for jdx in range(N):
            sum_ += _gaussian_kernel(samples1[idx], samples2[jdx])

    return sum_ / (M * N)


@nb.njit(cache=True)
def gaussian_mmd(samples1, samples2):
    M = samples1.shape[0]
    N = samples2.shape[0]

    term1 = _get_kernel_expectation_value(samples1, M)
    term2 = _get_kernel_expectation_value(samples2, N)

    cross_term = _get_cross_term(samples1, samples2, M, N)

    return term1 + term2 - 2 * cross_term


@nb.njit(cache=True)
def gaussian_mmd_grad(a, b, x, y):
    size_a = a.shape[0]
    size_b = b.shape[0]
    size_x = x.shape[0]
    size_y = y.shape[0]

    return (
        _get_cross_term(a, x, size_a, size_x)
        - _get_cross_term(b, x, size_b, size_x)
        - _get_cross_term(a, y, size_a, size_y)
        + _get_cross_term(b, y, size_b, size_y)
    )


class AdamOptimizer:
    def __init__(
        self,
        learning_rate: float = 1e-03,
        decay1: float = 0.9,
        decay2: float = 0.999,
        number_of_weights: int = 1,
    ):
        self.learning_rate = learning_rate
        self.decay1 = decay1
        self.decay2 = decay2
        self.first_moment = np.zeros(number_of_weights, dtype=float)
        self.second_moment = np.zeros(number_of_weights, dtype=float)
        self.step_count = 0

        self.epsilon = 1e-08

    def update_weights(self, weights: np.ndarray, gradient: np.ndarray):
        self.step_count += 1

        self.first_moment = (
            self.decay1 * self.first_moment + (1 - self.decay1) * gradient
        )
        self.second_moment = self.decay2 * self.second_moment + (1 - self.decay2) * (
            np.square(gradient)
        )

        current_learning_rate = (
            self.learning_rate
            * (np.sqrt(1 - self.decay2**self.step_count))
            / (1 - self.decay1**self.step_count)
        )

        return weights - current_learning_rate * self.first_moment / (
            np.sqrt(self.second_moment) + self.epsilon
        )
