def test_ndscan_results_module_imports_without_ndscan():
    from dnamic_toolkit import ndscan_results

    assert ndscan_results.LabNdscanRun is not None
