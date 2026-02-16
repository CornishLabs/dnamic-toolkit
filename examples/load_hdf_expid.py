from dnamic_toolkit.results import load_hdf5_file
import numpy as np
import matplotlib.pyplot as plt

fn = "/home/lab/artiq-files/dnamic-lab/results/2026-01-12/18/000000877-LoadRbMOT.h5"   # <-- your file
r = load_hdf5_file(fn)

expid = r["expid"]

print(type(expid["arguments"]))
print()
print(list(expid["arguments"].keys()))
print()
print(expid["arguments"]["ndscan_params"])