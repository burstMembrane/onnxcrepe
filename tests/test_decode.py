import numpy as np
import crepetrt


###############################################################################
# Test decode.py
###############################################################################


def test_weighted_argmax_decode():
    """Tests that weighted argmax decode works without CUDA assertion error"""
    fake_logits = np.random.rand(8, 360, 128)
    decoded = crepetrt.decode.weighted_argmax(fake_logits)
