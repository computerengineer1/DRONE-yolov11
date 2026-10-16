"""
Drone Reconnaissance & Targeting System
========================================
Military-grade AI detection with YOLOv11, Kalman tracking, and Cybernetic HUD.
Optimized for NVIDIA CUDA / TensorRT acceleration.

Usage:
    python main.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
from ui.app import DroneReconApp


def main():
    # Force dark mode
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    # Launch application
    app = DroneReconApp()
    app.mainloop()


if __name__ == "__main__":
    main()
