import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

with open("ggh_errors.json", "r") as f:
    data = json.load(f)

errors = np.array(data["errors"])
b1 = np.array(data["basis_sk"][0])
b2 = np.array(data["basis_sk"][1])

v1 = -0.5 * b1 - 0.5 * b2
v2 =  0.5 * b1 - 0.5 * b2
v3 =  0.5 * b1 + 0.5 * b2
v4 = -0.5 * b1 + 0.5 * b2
poly_corners = np.array([v1, v2, v3, v4])

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(errors[:, 0], errors[:, 1], s=4, alpha=0.25, color="crimson", label="Вектори помилок $e$")

parallelogram = Polygon(poly_corners, closed=True, fill=False, edgecolor="black", linewidth=2, linestyle="--", label="Межі $\mathcal{P}_{1/2}(B_{sk})$")
ax.add_patch(parallelogram)

ax.quiver(0, 0, b1[0], b1[1], angles='xy', scale_units='xy', scale=1, color='blue', label="$b_1$")
ax.quiver(0, 0, b2[0], b2[1], angles='xy', scale_units='xy', scale=1, color='green', label="$b_2$")

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.6)
ax.set_title("Secret key leak in GGH (Rust simulation)")
ax.legend(loc="upper right")

plt.show()
