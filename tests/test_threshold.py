import numpy as np

import crepetrt


###############################################################################
# Test threshold.py
###############################################################################


def test_at():
    """Test crepetrt.threshold.At"""
    input_pitch = np.array([100., 110., 120., 130., 140.])
    periodicity = np.array([.19, .22, .25, .17, .30])

    # Perform thresholding
    output_pitch = crepetrt.threshold.At(.20)(input_pitch, periodicity)

    # Ensure thresholding is not in-place
    assert not (input_pitch == output_pitch).all()

    # Ensure certain frames are marked as unvoiced
    isnan = np.isnan(output_pitch)
    assert isnan[0] and isnan[3]
    assert not isnan[1] and not isnan[2] and not isnan[4]
