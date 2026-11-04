import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HOUSE_COLORS = plt.get_cmap("tab20").colors


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    })
