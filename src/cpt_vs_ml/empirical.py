"""Equations transcribed from the local notebook; original units must be checked.

These functions have no verified physical-unit contract and are not a validated
engineering predictor. The CLI does not compare them on arbitrary synthetic units.
"""
def depth_qt(depth, qt): return 35*depth**0.032*qt**0.284-121.6
def qt_only(qt): return 42.52*qt**0.2785-138.8
def qt_u2(qt, u2): return 65.91*qt**0.248*u2**0.0262-204.8
def qt_fs(qt, fs): return 1251*qt**0.44*fs**0.0027-1506
