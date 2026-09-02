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
