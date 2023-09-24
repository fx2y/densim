import unittest

import numpy as np

from memory_storage import MemoryStorage


class TestMemoryStorage(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        self.memory_storage = MemoryStorage(self.vectors)

    def test_getitem(self):
        self.assertTrue(np.array_equal(self.memory_storage[0], np.array([1, 2, 3], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.memory_storage[1], np.array([4, 5, 6], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.memory_storage[2], np.array([7, 8, 9], dtype=np.float32)))

    def test_setitem(self):
        self.memory_storage[0] = np.array([10, 11, 12], dtype=np.float32)
        self.assertTrue(np.array_equal(self.memory_storage[0], np.array([10, 11, 12], dtype=np.float32)))

    def test_len(self):
        self.assertEqual(len(self.memory_storage), 3)

    def test_add(self):
        self.memory_storage.add(np.array([10, 11, 12], dtype=np.float32))
        self.assertEqual(len(self.memory_storage), 4)
        self.assertTrue(np.array_equal(self.memory_storage[3], np.array([10, 11, 12], dtype=np.float32)))

    def test_delete(self):
        self.memory_storage.delete(0)
        self.assertEqual(len(self.memory_storage), 2)
        self.assertTrue(np.array_equal(self.memory_storage[0], np.array([4, 5, 6], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.memory_storage[1], np.array([7, 8, 9], dtype=np.float32)))

    def test_euclidean_distance(self):
        self.assertAlmostEqual(self.memory_storage.euclidean_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                      np.array([4, 5, 6], dtype=np.float32)), 5.196152,
                               places=6)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(self.memory_storage.manhattan_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                      np.array([4, 5, 6], dtype=np.float32)), 9.0,
                               places=6)

    def test_chebyshev_distance(self):
        self.assertAlmostEqual(self.memory_storage.chebyshev_distance(np.array([1, 2, 3], dtype=np.float32),
                                                                      np.array([4, 5, 6], dtype=np.float32)), 3.0,
                               places=6)
