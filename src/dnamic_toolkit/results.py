import h5py
import json

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

def load_hdf5_file(filename: str) -> Dict[str, Any]:
    """Load an ARTIQ results file. (copy of oitg func)

    :returns: A dictionary containing the logical contents of the HDF5 file, including:

     * ``"start_time"``: the Unix timestamp when the experiment was built
     * ``"expid"``: experiment description, including submission arguments
     * ``"datasets"``: dictionary containing all set datasets
    """
    with h5py.File(filename, "r") as f:
        r = {}

        # `expid` is actually serialised as PYON on the ARTIQ side, but as it is within
        # the JSON subset (as of ARTIQ 5, anyway), we can avoid the library dependency
        # in data analysis environments.
        r["expid"] = json.loads(f["expid"][()])

        for k in ["artiq_version", "start_time"]:
            r[k] = f[k][()]

        ds = {}
        r["datasets"] = ds
        for k in f["datasets"]:
            ds[k] = f["datasets"][k][()]

        ar = {}
        r["archive"] = ar
        for k in f["archive"]:
            ar[k] = f["archive"][k][()]

        return r