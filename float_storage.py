import numpy as np


class FloatStorage:
    def __init__(self, vectors):
        self.vectors = np.array(vectors, dtype=np.float32)

    def __getitem__(self, index):
        return self.vectors[index]

    def __setitem__(self, index, value):
        self.vectors[index] = value

    def __len__(self):
        return len(self.vectors)

    def add(self, vector):
        self.vectors = np.append(self.vectors, [vector], axis=0)

    def delete(self, index):
        self.vectors = np.delete(self.vectors, index, axis=0)

    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def manhattan_distance(self, vector1, vector2):
        return np.sum(np.abs(vector1 - vector2))

    def chebyshev_distance(self, vector1, vector2):
        return np.max(np.abs(vector1 - vector2))

    def add_vectors(self, vector1, vector2):
        return vector1 + vector2

    def subtract_vectors(self, vector1, vector2):
        return vector1 - vector2

    def multiply_vectors(self, vector1, vector2):
        return vector1 * vector2

    def divide_vectors(self, vector1, vector2):
        return vector1 / vector2
