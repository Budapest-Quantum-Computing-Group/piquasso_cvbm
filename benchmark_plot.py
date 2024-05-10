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

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
mpl.rc("text", usetex=True)

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        data = json.load(f)

    plt.scatter(data["range"], data["means"], marker="x")

    plt.yticks(data["range"])

    plt.yscale("log")

    x = data["range"]
    y = np.log(data["means"])

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    x = np.linspace(min(data["range"]), max(data["range"]))

    y = np.exp(m * x + c)

    np.set_printoptions(precision=4)

    plt.plot(x, y, label=f"$y = \exp({str(m)[:6]}x{str(c)[:7]})$", color="red")
    plt.legend()
    plt.xlabel("Number of modes")
    plt.ylabel("Runtime")

    plt.xticks(data["range"])

    plt.show()
