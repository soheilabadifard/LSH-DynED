import numpy as np
np.random.seed(101)


class EnhancedLSH:
    def __init__(self, n_dimes, n_hypers):
        self.n_dimensions = n_dimes
        self.n_hyperplanes = n_hypers
        # Initialize random hyperplanes and transformation matrix
        self.hyperplanes = np.random.randn(n_hypers, n_dimes)
        self.transformation_matrix = np.random.randn(n_dimes, n_dimes)
        self.thresholds = np.zeros(n_hypers)
        self.hash_table = {}

    def fit(self, data):
        # array = convert_to_array(data)
        # Apply random transformation (rotation)
        rotated_data = self._apply_rotation(data)
        # Train thresholds based on rotated data
        self._train_thresholds(rotated_data)

    def _apply_rotation(self, data):
        # Apply a random linear transformation (rotation)
        return np.dot(data, self.transformation_matrix.T)

    def _train_thresholds(self, data):
        # Set thresholds as median of each dimension of the rotated data
        projected_data = np.dot(data, self.hyperplanes.T)
        self.thresholds = np.mean(projected_data, axis=0)

    def _hash(self, q_point):
        rotated_point = np.dot(q_point, self.transformation_matrix.T)
        projected_point = np.dot(rotated_point, self.hyperplanes.T)
        # Hash based on whether the projected point is above or below the threshold
        hash_code = ''.join(['1' if i > t else '0' for i, t in zip(projected_point, self.thresholds)])
        return hash_code

    def add_point(self, q_point_id, q_input):
        hash_code = self._hash(q_input)
        if hash_code in self.hash_table:
            self.hash_table[hash_code].add(q_point_id)
        else:
            self.hash_table[hash_code] = {q_point_id}

    def query(self, q_input, max_results=10):
        hash_code = self._hash(q_input)
        return list(self.hash_table.get(hash_code, []))[:max_results]

    def hamming_distance(self, hash1, hash2):
        """Calculate the Hamming distance between two hash codes."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def h_query(self, q_input, max_results=10):
        query_hash = self._hash(q_input)
        candidates = []
        for hash_code, ids in self.hash_table.items():
            distance = self.hamming_distance(query_hash, hash_code)
            for id1 in ids:
                candidates.append((distance, id1))
        # Sort candidates based on Hamming distance
        candidates.sort()
        # Return the IDs of the closest matches
        return [id1 for _, id1 in candidates[:max_results]]

    def get_hash_table(self):
        return self.hash_table

    def get_bucket_ids(self, bucket_id):
        return self.hash_table.get(bucket_id, [])


"""# Example Usage
n_dimensions = 128
n_hyperplanes = 5
lsh = EnhancedLSH(n_dimensions, n_hyperplanes)

# Randomly generated dataset and query (for demonstration purposes)
data = np.random.random((1000, n_dimensions)).astype('float32')
query = np.random.random((1, n_dimensions)).astype('float32')

# Assuming 'data' is your dataset as a numpy array of shape (n_samples, n_dimensions)
lsh.fit(data)

# Adding points to the LSH
for point_id, point in enumerate(data):
    lsh.add_point(point_id, point)

# Querying
query_point = np.random.randn(n_dimensions)
result_ids = lsh.query(query_point)
h_result_ids = lsh.h_query(query_point)
print("Nearest neighbors:", result_ids)
print("Nearest neighbors (Hamming):", h_result_ids)"""
