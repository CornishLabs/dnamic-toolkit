import numpy as np
import pytest

from dnamic_toolkit.calibrations.cs_1066_power import (
    CS_1066_TEST_POINT_POWER_CALIBRATIONS,
    CS_1066_TEST_POINT_POWER_RID_8232,
    CURRENT_CS_1066_TEST_POINT_POWER,
    cs_1066_test_point_power_w,
)


def test_current_calibration_is_traceable_record():
    assert CURRENT_CS_1066_TEST_POINT_POWER is CS_1066_TEST_POINT_POWER_RID_8232
    assert CURRENT_CS_1066_TEST_POINT_POWER.source_rid == 8232
    assert CURRENT_CS_1066_TEST_POINT_POWER.valid_input_range == (0.01, 7.5)
    assert CS_1066_TEST_POINT_POWER_CALIBRATIONS == (
        CURRENT_CS_1066_TEST_POINT_POWER,
    )


def test_scalar_setpoint_returns_power_in_watts():
    assert cs_1066_test_point_power_w(5.65) == pytest.approx(
        0.14767824386100453
    )


def test_array_setpoints_are_supported_for_analysis():
    result = cs_1066_test_point_power_w([1.0, 5.65, 7.5])

    assert isinstance(result, np.ndarray)
    assert result == pytest.approx(
        [0.008609552999999999, 0.14767824386100453, 0.5874276459400006]
    )
