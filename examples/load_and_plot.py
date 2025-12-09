# from oitg.results import load_hdf5_file
from dnamic_toolkit.results import load_hdf5_file
import numpy as np

fn = "/home/lab/dnamic-lab/results/2025-12-09/15/000000492-RabiFlopWithAnalysis.h5"   # <-- your file
r = load_hdf5_file(fn)

print("expid:", r["expid"])
print("artiq_version:", r.get("artiq_version"))
print("start_time:", r.get("start_time"))

print("\nDatasets:")
for k, v in r["datasets"].items():
    arr = np.asarray(v)
    print(f"  {k:30s}  shape={arr.shape} dtype={arr.dtype}")

print("\nArchive entries:")
for k, v in r["archive"].items():
    # archive items are often bytes/blobs
    print(f"  {k:30s}  type={type(v)}  len={len(v) if hasattr(v, '__len__') else '??'}")

