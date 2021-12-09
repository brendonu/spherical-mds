
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from orix import vector


# We'll want our plots to look a bit larger than the default size
new_params = {
    "figure.figsize": (10, 5),
    "lines.markersize": 10,
    "font.size": 20,
    "axes.grid": False,
}
plt.rcParams.update(new_params)
v1 = vector.Vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
plt.rcParams["axes.grid"] = True

v1.scatter()

v2 = vector.Vector3d([[1, 1, 2], [1, 1, -1]])


plt.clf()

fig, ax = plt.subplots(
    figsize=(5, 5), subplot_kw=dict(projection="stereographic")
)
azimuth = np.deg2rad([0, 60, 180])
polar = np.deg2rad([0, 60, 60])
print(azimuth)
print(polar)
ax.scatter(azimuth, polar, c=["C0", "C1", "C2"], s=200)
plt.savefig('test.png')

plt.clf()

v6 = vector.Vector3d.from_polar(
    azimuth=np.deg2rad([0, 60]), polar=np.deg2rad([0, 60]),
)

colors = ["C0", "C1", "C2"]
fig6 = v6.scatter(
    c=colors,
    s=200,
    axes_labels=["RD", "TD", None],
    show_hemisphere_label=True,
    return_figure=True
)
print(v6)
v6.draw_circle(color=colors, linewidth=2, figure=fig6)
plt.savefig('test.png')
