"""Cs 1066 nm tweezer setpoint-to-power calibrations."""

from .models import RoundedTwoSlopeCalibration


# This fit used the power-meter test point and 8 dB SU-Servo DDS attenuation from
# ``SUServo1066Powermeter``. The measured scan covered 0.01--7.5 V, with four
# repeats at each of 150 setpoints. Keep this named record unchanged when a new
# calibration is made: append another dated record and move the ``CURRENT`` alias.
CS_1066_TEST_POINT_POWER_RID_8232 = RoundedTwoSlopeCalibration(
    identifier="cs-1066-test-point-power-2026-07-17-rid-8232",
    source_rid=8232,
    measured_on="2026-07-17",
    offset=0.000379573,
    slope_low=0.00822998,
    slope_change=0.229497,
    transition=5.21098,
    width=0.0741287,
    valid_input_range=(0.01, 7.5),
    input_unit="V",
    output_unit="W",
    notes="1066 nm power-meter test point; SU-Servo DDS attenuation 8 dB.",
)

# Append future immutable records here so they are easy to discover interactively.
CS_1066_TEST_POINT_POWER_CALIBRATIONS = (
    CS_1066_TEST_POINT_POWER_RID_8232,
)

CURRENT_CS_1066_TEST_POINT_POWER = CS_1066_TEST_POINT_POWER_RID_8232


def cs_1066_test_point_power_w(setpoint_v, calibration=None):
    """Convert Cs tweezer servo setpoint voltage to test-point power in watts.

    Passing an explicit historical record keeps old analyses reproducible. Omitting
    it uses the calibration currently selected above.
    """
    if calibration is None:
        calibration = CURRENT_CS_1066_TEST_POINT_POWER
    return calibration.evaluate(setpoint_v)
