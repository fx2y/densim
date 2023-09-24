import numpy as np


class ByteStorage:
    def __init__(self, vectors, bits_per_vector):
        self.bits_per_vector = bits_per_vector
        self.min_val = np.min(vectors)
        self.max_val = np.max(vectors)
        self.vectors = self.quantize(vectors)

    def __getitem__(self, index):
        return self.dequantize(self.vectors[index])

    def __setitem__(self, index, value):
        self.vectors[index] = self.quantize(value)

    def __len__(self):
        return len(self.vectors)

    def quantize(self, vector):
        # Use uniform quantization to convert floats to bytes
        # Scale the values to the range [0, 255] and round to the nearest integer
        # Then cast to uint8 to get an 8-bit integer
        return np.round((vector - self.min_val) / (self.max_val - self.min_val) * 255).astype(np.uint8)

    def dequantize(self, vector):
        # Convert bytes to floats by reversing the quantization process
        # Cast to float32 to get a float
        return (vector / 255 * (self.max_val - self.min_val) + self.min_val).astype(np.float32)

    def add(self, vector):
        self.vectors = np.append(self.vectors, [self.quantize(vector)], axis=0)

    def delete(self, index):
        self.vectors = np.delete(self.vectors, index, axis=0)

    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(self.quantize(vector1) - self.quantize(vector2)) * (self.max_val - self.min_val) / 255

    def manhattan_distance(self, vector1, vector2):
        return np.sum(np.abs(self.quantize(vector1) - self.quantize(vector2))) * (self.max_val - self.min_val) / 255

    def chebyshev_distance(self, vector1, vector2):
        return np.max(np.abs(self.quantize(vector1) - self.quantize(vector2))) * (self.max_val - self.min_val) / 255
