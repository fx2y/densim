import unittest

import numpy as np

from float_storage import FloatStorage


class TestFloatStorage(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        self.float_storage = FloatStorage(self.vectors)

    def test_getitem(self):
        self.assertTrue(np.array_equal(self.float_storage[0], np.array([1, 2, 3], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.float_storage[1], np.array([4, 5, 6], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.float_storage[2], np.array([7, 8, 9], dtype=np.float32)))

    def test_setitem(self):
        self.float_storage[0] = np.array([10, 11, 12], dtype=np.float32)
        self.assertTrue(np.array_equal(self.float_storage[0], np.array([10, 11, 12], dtype=np.float32)))

    def test_len(self):
        self.assertEqual(len(self.float_storage), 3)

    def test_add(self):
        self.float_storage.add(np.array([10, 11, 12], dtype=np.float32))
        self.assertEqual(len(self.float_storage), 4)
        self.assertTrue(np.array_equal(self.float_storage[3], np.array([10, 11, 12], dtype=np.float32)))

    def test_delete(self):
        self.float_storage.delete(0)
        self.assertEqual(len(self.float_storage), 2)
        self.assertTrue(np.array_equal(self.float_storage[0], np.array([4, 5, 6], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.float_storage[1], np.array([7, 8, 9], dtype=np.float32)))

    def test_euclidean_distance(self):
        self.assertAlmostEqual(self.float_storage.euclidean_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                     np.array([4, 5, 6], dtype=np.float32)), 5.196152,
                               places=6)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(self.float_storage.manhattan_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                     np.array([4, 5, 6], dtype=np.float32)), 9.0,
                               places=6)

    def test_chebyshev_distance(self):
        self.assertAlmostEqual(self.float_storage.chebyshev_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                     np.array([4, 5, 6], dtype=np.float32)), 3.0,
                               places=6)

    def test_add_vectors(self):
        self.assertTrue(np.array_equal(self.float_storage.add_vectors(np.array([1, 2, 3], dtype=np.float32),
                                                                      np.array([4, 5, 6], dtype=np.float32)),
                                       np.array([5, 7, 9], dtype=np.float32)))

    def test_subtract_vectors(self):
        self.assertTrue(np.array_equal(self.float_storage.subtract_vectors(np.array([4, 5, 6], dtype=np.float32),
                                                                           np.array([1, 2, 3], dtype=np.float32)),
                                       np.array([3, 3, 3], dtype=np.float32)))

    def test_multiply_vectors(self):
        self.assertTrue(np.array_equal(self.float_storage.multiply_vectors(np.array([1, 2, 3], dtype=np.float32),
                                                                           np.array([4, 5, 6], dtype=np.float32)),
                                       np.array([4, 10, 18], dtype=np.float32)))

    def test_divide_vectors(self):
        self.assertTrue(np.array_equal(self.float_storage.divide_vectors(np.array([4, 5, 6], dtype=np.float32),
                                                                         np.array([1, 2, 3], dtype=np.float32)),
                                       np.array([4, 2.5, 2], dtype=np.float32)))
