import unittest

import numpy as np

from byte_storage import ByteStorage


class TestByteStorage(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        self.bits_per_vector = 8
        self.byte_storage = ByteStorage(self.vectors, self.bits_per_vector)
        self.rtol = 1.e-2

    def test_getitem(self):
        self.assertTrue(np.allclose(self.byte_storage[0], np.array([1, 2, 3], dtype=np.float32), self.rtol))
        self.assertTrue(np.allclose(self.byte_storage[1], np.array([4, 5, 6], dtype=np.float32), self.rtol))
        self.assertTrue(np.allclose(self.byte_storage[2], np.array([7, 8, 9], dtype=np.float32), self.rtol))

    def test_setitem(self):
        self.byte_storage[0] = np.array([2, 5, 8], dtype=np.float32)
        self.assertTrue(np.allclose(self.byte_storage[0], np.array([2, 5, 8], dtype=np.float32), self.rtol))

    def test_len(self):
        self.assertEqual(len(self.byte_storage), 3)

    def test_add(self):
        self.byte_storage.add(np.array([2, 5, 8], dtype=np.float32))
        self.assertEqual(len(self.byte_storage), 4)
        self.assertTrue(np.allclose(self.byte_storage[3], np.array([2, 5, 8], dtype=np.float32), self.rtol))

    def test_delete(self):
        self.byte_storage.delete(0)
        self.assertEqual(len(self.byte_storage), 2)
        self.assertTrue(np.allclose(self.byte_storage[0], np.array([4, 5, 6], dtype=np.float32), self.rtol))
        self.assertTrue(np.allclose(self.byte_storage[1], np.array([7, 8, 9], dtype=np.float32), self.rtol))

    def test_quantize(self):
        self.assertTrue(np.array_equal(self.byte_storage.quantize(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
                                       np.array([0, 32, 64], dtype=np.uint8)))

    def test_dequantize(self):
        self.assertTrue(np.allclose(self.byte_storage.dequantize(np.array([0, 32, 64], dtype=np.uint8)),
                                    np.array([1.0, 2.0, 3.0], dtype=np.float32), self.rtol))

    def test_euclidean_distance(self):
        self.assertAlmostEqual(self.byte_storage.euclidean_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                    np.array([4, 5, 6], dtype=np.float32)), 8.712366424785127,
                               places=6)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(self.byte_storage.manhattan_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                    np.array([4, 5, 6], dtype=np.float32)), 15.090196078431372,
                               places=6)

    def test_chebyshev_distance(self):
        self.assertAlmostEqual(self.byte_storage.chebyshev_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                    np.array([4, 5, 6], dtype=np.float32)), 5.050980392156863,
                               places=6)
