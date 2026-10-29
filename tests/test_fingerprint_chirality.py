import numpy as np
import pytest

from data_process import MACCS_BITS, MIXFP_DIM, RDK_BITS, get_mix_fingerprint


@pytest.mark.parametrize("radius", [2, 3])
def test_morgan_fingerprint_distinguishes_enantiomers(radius):
    first = get_mix_fingerprint("N[C@@H](C)C(=O)O", radius=radius)
    mirror = get_mix_fingerprint("N[C@H](C)C(=O)O", radius=radius)
    reordered = get_mix_fingerprint("C[C@H](N)C(=O)O", radius=radius)
    assert first.shape == mirror.shape == (MIXFP_DIM,)
    morgan_start = MACCS_BITS + RDK_BITS
    assert not np.array_equal(first[morgan_start:], mirror[morgan_start:])
    np.testing.assert_array_equal(first, reordered)
