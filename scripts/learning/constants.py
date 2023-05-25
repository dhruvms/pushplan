import numpy as np
import torch
from matplotlib.path import Path

TABLE = np.array([0.78, -0.5, 0.75])
TABLE_SIZE = np.array([0.3, 0.4, 0.02])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CODES = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
