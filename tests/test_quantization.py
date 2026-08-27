import numpy as np

from sabir_tools.model_format import dequantize_q4, quantize_q4


def test_q4_round_trip_and_nibble_range():
    rng = np.random.default_rng(42)
    source = rng.normal(0, 0.25, size=(7, 130)).astype(np.float32)
    codes, scales, expected = quantize_q4(source, 64)
    actual = dequantize_q4(codes, scales, source.shape, 64)
    assert np.array_equal(actual, expected)
    packed = np.frombuffer(codes, dtype=np.uint8)
    assert np.all((packed & 0x0F) >= 1)
    assert np.all((packed >> 4) <= 15)
    assert np.sqrt(np.mean((actual - source) ** 2)) < 0.04


def test_q4_zero_tensor_is_stable():
    source = np.zeros((2, 64), dtype=np.float32)
    codes, scales, expected = quantize_q4(source, 64)
    assert np.all(dequantize_q4(codes, scales, source.shape, 64) == 0)
    assert np.all(expected == 0)

