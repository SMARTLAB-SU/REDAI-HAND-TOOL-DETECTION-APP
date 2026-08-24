"""
REDAI Hand Tool Detection Application - Main GUI Module

This module initializes the PyQt5 desktop user interface, connects to the Intel 
RealSense camera feed, runs real-time YOLO hand tool object detection, and performs 
ground-truth tool inventory comparison.
"""

import sys
import os
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    print("REDAI Hand Tool Detection Application initialized.")
