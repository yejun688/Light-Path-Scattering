import cv2
from .imports import *
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def show(frame: ArrayLike):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    show_ax(ax, frame)


def show_ax(ax, frame: ArrayLike):
    ax.imshow(frame**(1.0/2.2), vmin=0.0, vmax=1.0, cmap='gray')


def write(filename: str, frame: ArrayLike):
    cv2.imwrite(filename, np.array(frame))
