"""Lightweight console-script entry points for optional tools."""


def beam_profile():
    """Run the beam profiler, reporting its optional dependency group."""

    try:
        from .beam_profile import main
    except ModuleNotFoundError as error:
        if error.name in {"matplotlib", "tifffile"}:
            raise SystemExit(
                "dnamic-beam-profile requires dnamic-toolkit[beam-profile]"
            ) from error
        raise
    return main()


def thorlabs_camera():
    """Run the Thorlabs live viewer, reporting its optional dependency group."""

    try:
        from .thorlabs_camera_viewer import main
    except ModuleNotFoundError as error:
        if error.name in {"PyQt6", "pyqtgraph"}:
            raise SystemExit(
                "dnamic-thorlabs-camera requires "
                "dnamic-toolkit[camera-viewer] and thorlabs-tsi-sdk"
            ) from error
        raise
    return main()
