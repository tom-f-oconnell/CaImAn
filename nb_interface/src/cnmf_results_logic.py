import bqplot
from bqplot import (
	LogScale, LinearScale, OrdinalColorScale, ColorAxis,
	Axis, Scatter, Lines, CATEGORY10, Label, Figure, Tooltip, Toolbar
)
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, VBox, IntSlider, Play, jslink, Tab
import ipywidgets as widgets
import os

def load_context_data(context):
	try:
		est = context.cnm.estimates
		A, C, b, f, YrA, sn, conv = est.A, est.C, est.b, est.f, est.YrA, est.sn, est.S
		return A, C, b, f, YrA, sn, conv
	except Exception as e:
		print("Error loading context.")
		return None

#Hold's contours
contours = None
