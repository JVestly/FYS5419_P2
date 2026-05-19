"""
This file contains utilities, i.e. most functions and classes for the project.
"""
from __future__ import annotations
import random
import math
import itertools
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from collections import defaultdict
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector


class TSP:
    """Generate and solve small Euclidean TSP instances."""

    def __init__(self, n_locations: int = 10, seed: int | None = None):
        if n_locations <= 0:
            raise ValueError("n_locations must be a positive integer.")

        self.n_locations = n_locations
        self.seed = seed
        self.map: dict[int, tuple[float, float]] = {}
        self.distance_matrix: list[list[float]] = []

    def create_map(
        self,
        x_range: tuple[float, float] = (0, 100),
        y_range: tuple[float, float] = (0, 100),
    ) -> dict[int, tuple[float, float]]:
        """Create a dictionary mapping location indices to random (x, y) coordinates."""
        rng = random.Random(self.seed)

        self.map = {
            i: (
                rng.uniform(*x_range),
                rng.uniform(*y_range),
            )
            for i in range(self.n_locations)
        }

        return self.map

    def compute_distance_matrix(self) -> list[list[float]]:
        """Compute the Euclidean distance matrix between all location pairs."""
        if not self.map:
            raise ValueError("Map has not been created yet. Call create_map() first.")

        self.distance_matrix = [
            [
                math.dist(self.map[i], self.map[j])
                for j in range(self.n_locations)
            ]
            for i in range(self.n_locations)
        ]

        return self.distance_matrix

    def is_valid_tour(self, tour: list[int]) -> bool:
        """Check whether a tour is a Hamiltonian cycle."""
        if len(tour) != self.n_locations + 1:
            return False

        if tour[0] != tour[-1]:
            return False

        visited = tour[:-1]
        return sorted(visited) == list(range(self.n_locations))

    def tour_cost(self, tour: list[int]) -> float:
        """Compute the total distance of a tour."""
        if not self.distance_matrix:
            raise ValueError("Distance matrix has not been computed yet.")

        if not self.is_valid_tour(tour):
            raise ValueError("Invalid tour. Tour must be a Hamiltonian cycle.")

        return sum(
            self.distance_matrix[tour[i]][tour[i + 1]]
            for i in range(len(tour) - 1)
        )

    def random_tour(self, start: int = 0) -> list[int]:
        """Generate a random valid tour starting and ending at start."""
        self._check_start(start)

        rng = random.Random(self.seed)

        cities = list(range(self.n_locations))
        cities.remove(start)
        rng.shuffle(cities)

        return [start] + cities + [start]

    def greedy_search(self, start: int = 0) -> list[int]:
        """Construct a tour using nearest-neighbor greedy search."""
        self._check_ready()
        self._check_start(start)

        unvisited = set(range(self.n_locations))
        unvisited.remove(start)

        tour = [start]
        current = start

        while unvisited:
            next_city = min(
                unvisited,
                key=lambda city: self.distance_matrix[current][city],
            )

            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        tour.append(start)
        return tour

    def two_opt(self, initial_tour: list[int] | None = None) -> tuple[list[int], float]:
        """
        Improve a tour using 2-opt local search.

        This is the TSP analogue of a simple local descent method:
        repeatedly reverse parts of the tour whenever that lowers the cost.
        """
        self._check_ready()

        if initial_tour is None:
            tour = self.greedy_search(start=0)
        else:
            if not self.is_valid_tour(initial_tour):
                raise ValueError("Initial tour is not valid.")
            tour = initial_tour[:]

        best_cost = self.tour_cost(tour)
        improved = True

        while improved:
            improved = False

            for i in range(1, self.n_locations - 1):
                for j in range(i + 1, self.n_locations):
                    if j - i == 1:
                        continue

                    new_tour = tour[:]
                    new_tour[i:j] = reversed(new_tour[i:j])

                    new_cost = self.tour_cost(new_tour)

                    if new_cost < best_cost:
                        tour = new_tour
                        best_cost = new_cost
                        improved = True

        return tour, best_cost


    def held_karp(self, start: int = 0) -> tuple[list[int], float]:
        """
        Exact dynamic-programming solver for small TSP instances.

        Time complexity is O(n^2 2^n), so use only for small n.
        """
        self._check_ready()
        self._check_start(start)

        cities = tuple(city for city in range(self.n_locations) if city != start)

        dp = {}

        for city in cities:
            dp[(frozenset([city]), city)] = (
                self.distance_matrix[start][city],
                [start, city],
            )

        for subset_size in range(2, len(cities) + 1):
            for subset in itertools.combinations(cities, subset_size):
                subset_set = frozenset(subset)

                for last in subset:
                    previous_set = subset_set - {last}

                    best_cost = float("inf")
                    best_path = []

                    for previous in previous_set:
                        previous_cost, previous_path = dp[(previous_set, previous)]
                        cost = previous_cost + self.distance_matrix[previous][last]

                        if cost < best_cost:
                            best_cost = cost
                            best_path = previous_path + [last]

                    dp[(subset_set, last)] = (best_cost, best_path)

        full_set = frozenset(cities)

        best_cost = float("inf")
        best_tour = []

        for last in cities:
            path_cost, path = dp[(full_set, last)]
            total_cost = path_cost + self.distance_matrix[last][start]

            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = path + [start]

        return best_tour, best_cost

    def _check_ready(self) -> None:
        """Check that the distance matrix has been computed."""
        if not self.distance_matrix:
            raise ValueError("Distance matrix has not been computed yet.")

    def _check_start(self, start: int) -> None:
        """Check that the start city is valid."""
        if start not in range(self.n_locations):
            raise ValueError("Start location is not valid.")




class QAOA:
    """
    Contains methods for the QAOA (Quantum Approximation Optimization Algorithm)
    """
    def __init__(
        self,
        tsp,
        layers=1,
        initial_route=None,
        seed=42,
        fixed_start=True,
        compressed_basis=False,
        neighborhood_depth=1,
        neighborhood_moves=("swap", "two_opt"),
        max_basis_size=None,
        pad_compressed_basis=True,
        compressed_basis_scope="local",
        mixer="grover",
    ):
        self.tsp = tsp
        self.n = tsp.n_locations
        self.fixed_start = fixed_start
        self.compressed_basis = compressed_basis
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_moves = tuple(neighborhood_moves)
        self.max_basis_size = max_basis_size
        self.pad_compressed_basis = pad_compressed_basis
        self.compressed_basis_scope = compressed_basis_scope
        self.mixer = str(mixer).lower()
        if self.mixer == "exchange":
            self.mixer = "grover"

        allowed_mixers = {"grover", "x", "xy"}
        if self.mixer not in allowed_mixers:
            raise ValueError(f"mixer must be one of {sorted(allowed_mixers)}, got {mixer!r}.")

        if fixed_start:
            self.active_n = self.n - 1
            self.num_qubits = self.active_n ** 2
        else:
            self.active_n = self.n
            self.num_qubits = self.n ** 2

        self.layers = layers
        self.initial_route = initial_route
        self.seed = seed

        rng = np.random.default_rng(seed)

        self.gamma = rng.uniform(0.0, 2.0 * np.pi, self.layers)
        self.beta = rng.uniform(0.0, np.pi, self.layers)


    def _to_ising(self, route):
        """Parameters
            route : 
        """

        x = np.asarray(route)

        if x.shape != (self.n, self.n):
            raise ValueError(f"Expected shape {(self.n, self.n)}, got {x.shape}.")

        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Input must be binary, containing only 0 and 1.")

        return 1 - 2 * x
    
        
    def _idx(self, i, t):
        """
        Map city i and position t to a qubit index.

        If fixed_start=True, city 0 at position 0 is removed.
        Then i,t are both in {1,...,n-1}.
        """
        if self.fixed_start:
            return (i - 1) * self.active_n + (t - 1)

        return i * self.n + t


    def _bitstring_to_one_hot(self, bitstring):
        if self.fixed_start:
            x = np.zeros((self.active_n, self.active_n), dtype=int)

            for i in range(1, self.n):
                for t in range(1, self.n):
                    q = self._idx(i, t)
                    x[i - 1, t - 1] = int(bitstring[self.num_qubits - 1 - q])

            return x

        x = np.zeros((self.n, self.n), dtype=int)

        for i in range(self.n):
            for t in range(self.n):
                q = self._idx(i, t)
                x[i, t] = int(bitstring[self.num_qubits - 1 - q])

        return x


    def _expand_one_hot(self, x):
        """
        Convert reduced fixed-start one-hot matrix to full n x n matrix.
        """
        if not self.fixed_start:
            return np.asarray(x)

        x = np.asarray(x)

        full = np.zeros((self.n, self.n), dtype=int)
        full[0, 0] = 1
        full[1:, 1:] = x

        return full


    def _one_hot_to_tour(self, one_hot_route):
        """
        Convert a valid one-hot route into the full tour [0, ..., 0].

        With fixed_start=True, one_hot_route is the reduced (n-1) x (n-1)
        encoding used by the quantum circuit.
        """
        full = self._expand_one_hot(one_hot_route)

        if full.shape != (self.n, self.n):
            raise ValueError(f"Expected full shape {(self.n, self.n)}, got {full.shape}.")

        route = [int(np.argmax(full[:, t])) for t in range(self.n)]
        route.append(route[0])

        return route


    def _tour_to_reduced_one_hot(self, tour):
        if not self.fixed_start:
            x = np.zeros((self.n, self.n), dtype=int)
            for t, city in enumerate(tour[:-1]):
                x[city, t] = 1
            return x

        if tour[0] != 0 or tour[-1] != 0:
            raise ValueError("Fixed-start tours must start and end at city 0.")

        x = np.zeros((self.active_n, self.active_n), dtype=int)

        for t, city in enumerate(tour[1:-1], start=1):
            x[city - 1, t - 1] = 1

        return x


    def _tour_key(self, tour):
        return tuple(int(city) for city in tour)


    def _swap_neighbors(self, tour):
        inner = list(tour[1:-1])
        neighbors = []

        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                candidate = inner.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                neighbors.append([0] + candidate + [0])

        return neighbors


    def _two_opt_neighbors(self, tour):
        inner = list(tour[1:-1])
        neighbors = []

        for i in range(len(inner) - 1):
            for j in range(i + 1, len(inner)):
                candidate = inner.copy()
                candidate[i:j + 1] = reversed(candidate[i:j + 1])
                neighbors.append([0] + candidate + [0])

        return neighbors


    def _route_neighbors(self, tour):
        neighbors = []

        if "swap" in self.neighborhood_moves:
            neighbors.extend(self._swap_neighbors(tour))

        if "two_opt" in self.neighborhood_moves:
            neighbors.extend(self._two_opt_neighbors(tour))

        return neighbors


    def build_compressed_basis(self):
        """
        Build a route-basis Hilbert space around the initial tour.

        Each basis vector is a valid full TSP tour, so the dimension is the
        number of selected tours instead of 2 ** ((n - 1) ** 2).
        """
        if not self.fixed_start:
            raise ValueError("Compressed basis currently assumes fixed_start=True.")

        if self.initial_route is None:
            center_tour = self.tsp.greedy_search(start=0)
        else:
            center_tour = self._one_hot_to_tour(self.initial_route)

        if self.compressed_basis_scope == "all_valid":
            routes = [
                [0] + list(perm) + [0]
                for perm in itertools.permutations(range(1, self.n))
            ]

            if self.max_basis_size is not None:
                routes = sorted(routes, key=self.tsp.tour_cost)[:self.max_basis_size]

            if self._tour_key(center_tour) not in {self._tour_key(route) for route in routes}:
                routes[0] = center_tour

        elif self.compressed_basis_scope == "local":
            routes = [center_tour]
            seen = {self._tour_key(center_tour)}
            frontier = [center_tour]

            for _ in range(self.neighborhood_depth):
                next_frontier = []

                for route in frontier:
                    for neighbor in self._route_neighbors(route):
                        key = self._tour_key(neighbor)
                        if key in seen:
                            continue

                        seen.add(key)
                        routes.append(neighbor)
                        next_frontier.append(neighbor)

                        if self.max_basis_size is not None and len(routes) >= self.max_basis_size:
                            break

                    if self.max_basis_size is not None and len(routes) >= self.max_basis_size:
                        break

                frontier = next_frontier

                if not frontier:
                    break

                if self.max_basis_size is not None and len(routes) >= self.max_basis_size:
                    break

        else:
            raise ValueError("compressed_basis_scope must be 'local' or 'all_valid'.")

        valid_dimension = len(routes)
        if self.pad_compressed_basis:
            dimension = 1 << int(np.ceil(np.log2(valid_dimension)))
        else:
            dimension = valid_dimension

        route_index = {self._tour_key(route): i for i, route in enumerate(routes)}
        valid_costs = np.array([self.tsp.tour_cost(route) for route in routes], dtype=float)
        invalid_cost = float(np.max(valid_costs) + self.n * np.max(self.tsp.distance_matrix))
        costs = np.full(dimension, invalid_cost, dtype=float)
        costs[:valid_dimension] = valid_costs

        mixer = np.zeros((dimension, dimension), dtype=float)

        for i, route in enumerate(routes):
            for neighbor in self._route_neighbors(route):
                j = route_index.get(self._tour_key(neighbor))
                if j is not None and i != j:
                    mixer[i, j] = 1.0

        mixer = np.maximum(mixer, mixer.T)

        if valid_dimension > 1:
            degrees = mixer.sum(axis=1)
            for i, degree in enumerate(degrees):
                if degree > 0:
                    mixer[i, i] = -degree

        eigvals, eigvecs = np.linalg.eigh(mixer)

        self.compressed_routes = routes
        self.compressed_costs = costs
        self.compressed_valid_costs = valid_costs
        self.compressed_mixer = mixer
        self.compressed_mixer_eigvals = eigvals
        self.compressed_mixer_eigvecs = eigvecs
        self.compressed_initial_index = route_index[self._tour_key(center_tour)]
        self.compressed_valid_dimension = valid_dimension
        self.compressed_dimension = dimension
        self.compressed_qubits = int(np.ceil(np.log2(dimension)))
        self.compressed_unused_states = dimension - valid_dimension

        return routes, costs, mixer
    
    
    
    def ansatz(self):
        qc = QuantumCircuit(self.num_qubits, name="ansatz")

        if self.initial_route is None:
            for q in range(self.num_qubits):
                qc.h(q)

        else:
            x = np.asarray(self.initial_route)

            expected_shape = (
                (self.active_n, self.active_n)
                if self.fixed_start
                else (self.n, self.n)
            )

            if x.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {x.shape}.")

            if not np.all((x == 0) | (x == 1)):
                raise ValueError("Route must be binary.")

            if self.fixed_start:
                for i in range(1, self.n):
                    for t in range(1, self.n):
                        if x[i - 1, t - 1] == 1:
                            qc.x(self._idx(i, t))
            else:
                for i in range(self.n):
                    for t in range(self.n):
                        if x[i, t] == 1:
                            qc.x(self._idx(i, t))

        self.ansatz_circuit = qc
        return qc


    def _classical_cost(self, one_hot_route, penalty_weight=None):
        x = np.asarray(one_hot_route)

        if not self.tsp.distance_matrix:
            raise ValueError("Distance matrix has not been computed.")

        d = np.asarray(self.tsp.distance_matrix)

        if penalty_weight is None:
            penalty_weight = self.n * np.max(d)

        if self.fixed_start:
            if x.shape != (self.active_n, self.active_n):
                raise ValueError(f"Expected shape {(self.active_n, self.active_n)}, got {x.shape}.")

            distance_cost = 0.0

            # Edge from fixed start city 0 to position 1
            for j in range(1, self.n):
                distance_cost += d[0, j] * x[j - 1, 0]

            # Middle edges
            for t in range(1, self.n - 1):
                for i in range(1, self.n):
                    for j in range(1, self.n):
                        distance_cost += d[i, j] * x[i - 1, t - 1] * x[j - 1, t]

            # Edge from last position back to city 0
            for i in range(1, self.n):
                distance_cost += d[i, 0] * x[i - 1, self.n - 2]

            city_penalty = sum(
                (1 - np.sum(x[i, :])) ** 2
                for i in range(self.active_n)
            )

            position_penalty = sum(
                (1 - np.sum(x[:, t])) ** 2
                for t in range(self.active_n)
            )

            return distance_cost + penalty_weight * (city_penalty + position_penalty)

        # Original full encoding
        if x.shape != (self.n, self.n):
            raise ValueError(f"Expected shape {(self.n, self.n)}, got {x.shape}.")

        distance_cost = 0.0

        for t in range(self.n):
            next_t = (t + 1) % self.n

            for i in range(self.n):
                for j in range(self.n):
                    distance_cost += d[i, j] * x[i, t] * x[j, next_t]

        city_penalty = sum((1 - np.sum(x[i, :])) ** 2 for i in range(self.n))
        position_penalty = sum((1 - np.sum(x[:, t])) ** 2 for t in range(self.n))

        return distance_cost + penalty_weight * (city_penalty + position_penalty)



    def _classical_cost_terms(self, penalty_weight=None):
        if not self.tsp.distance_matrix:
            raise ValueError("Distance matrix has not been computed.")

        w = np.asarray(self.tsp.distance_matrix)

        if penalty_weight is None:
            penalty_weight = self.n * np.max(w)

        linear = defaultdict(float)
        quadratic = defaultdict(float)
        offset = 0.0

        def add_quadratic(a, b, coeff):
            if a == b:
                linear[a] += coeff
            else:
                quadratic[tuple(sorted((a, b)))] += coeff

        if self.fixed_start:
            # Edge 0 -> first free position
            for j in range(1, self.n):
                linear[self._idx(j, 1)] += w[0, j]

            # Middle edges
            for t in range(1, self.n - 1):
                for i in range(1, self.n):
                    for j in range(1, self.n):
                        a = self._idx(i, t)
                        b = self._idx(j, t + 1)
                        add_quadratic(a, b, w[i, j])

            # Last free position -> 0
            for i in range(1, self.n):
                linear[self._idx(i, self.n - 1)] += w[i, 0]

            # Position constraints
            for t in range(1, self.n):
                variables = [self._idx(i, t) for i in range(1, self.n)]

                offset += penalty_weight

                for a in variables:
                    linear[a] -= penalty_weight

                for p in range(len(variables)):
                    for q in range(p + 1, len(variables)):
                        add_quadratic(variables[p], variables[q], 2 * penalty_weight)

            # City constraints
            for i in range(1, self.n):
                variables = [self._idx(i, t) for t in range(1, self.n)]

                offset += penalty_weight

                for a in variables:
                    linear[a] -= penalty_weight

                for p in range(len(variables)):
                    for q in range(p + 1, len(variables)):
                        add_quadratic(variables[p], variables[q], 2 * penalty_weight)

            return dict(linear), dict(quadratic), offset

        # Original full encoding
        for i in range(self.n):
            for j in range(self.n):
                for t in range(self.n):
                    a = self._idx(i, t)
                    b = self._idx(j, (t + 1) % self.n)
                    add_quadratic(a, b, w[i, j])

        for t in range(self.n):
            variables = [self._idx(i, t) for i in range(self.n)]

            offset += penalty_weight

            for a in variables:
                linear[a] -= penalty_weight

            for p in range(len(variables)):
                for q in range(p + 1, len(variables)):
                    add_quadratic(variables[p], variables[q], 2 * penalty_weight)

        for i in range(self.n):
            variables = [self._idx(i, t) for t in range(self.n)]

            offset += penalty_weight

            for a in variables:
                linear[a] -= penalty_weight

            for p in range(len(variables)):
                for q in range(p + 1, len(variables)):
                    add_quadratic(variables[p], variables[q], 2 * penalty_weight)

        return dict(linear), dict(quadratic), offset


    def _paulis(self):
        I2 = np.eye(2, dtype=complex)

        Zp = np.array(
            [[1, 0],
            [0, -1]],
            dtype=complex,
        )

        Xp = np.array(
            [[0, 1],
            [1, 0]],
            dtype=complex,
        )

        return I2, Xp, Zp


    def _pauli_y(self):
        return np.array(
            [[0, -1j],
            [1j, 0]],
            dtype=complex,
        )


    def _pauli_kron(self, P: np.ndarray, qubit: int) -> np.ndarray:
        I2, _, _ = self._paulis()

        ops = [I2] * self.num_qubits
        ops[qubit] = P

        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)

        return result


    def _two_pauli_kron(self, P: np.ndarray, q1: int, Q: np.ndarray, q2: int) -> np.ndarray:
        I2, _, _ = self._paulis()

        ops = [I2] * self.num_qubits
        ops[q1] = P
        ops[q2] = Q

        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)

        return result


    def _active_city_range(self):
        return range(1, self.n) if self.fixed_start else range(self.n)


    def _active_position_range(self):
        return range(1, self.n) if self.fixed_start else range(self.n)


    def _xy_mixer_pairs(self):
        pairs = set()
        cities = list(self._active_city_range())
        positions = list(self._active_position_range())

        for i in cities:
            for a, t in enumerate(positions):
                for u in positions[a + 1:]:
                    pairs.add(tuple(sorted((self._idx(i, t), self._idx(i, u)))))

        for t in positions:
            for a, i in enumerate(cities):
                for j in cities[a + 1:]:
                    pairs.add(tuple(sorted((self._idx(i, t), self._idx(j, t)))))

        return sorted(pairs)


    def build_HC(self, penalty_weight=None):
        linear, quadratic, offset = self._classical_cost_terms(penalty_weight)

        h = np.zeros(self.num_qubits)
        J = np.zeros((self.num_qubits, self.num_qubits))
        constant = offset

        for i, q_i in linear.items():
            constant += q_i / 2
            h[i] += -q_i / 2

        for (i, j), q_ij in quadratic.items():
            constant += q_ij / 4
            h[i] += -q_ij / 4
            h[j] += -q_ij / 4
            J[i, j] += q_ij / 4
            J[j, i] += q_ij / 4

        self.h = h
        self.J = J
        self.cost_offset = constant

        return h, J, constant


    def build_HM(self) -> np.ndarray:
        """
        Build the dense mixer Hamiltonian selected by self.mixer.

        mixer="x":
            H_M = sum_i X_i

        mixer="xy":
            H_M = 1/2 sum_(i,j) (X_i X_j + Y_i Y_j)

        mixer="grover":
            H_M = |s><s|, where |s> is the uniform superposition over
            the full computational basis.

        Returns:
            Dense matrix representation of H_M.
        """
        if self.num_qubits > 12:
            raise ValueError(
                "Dense mixer Hamiltonians are too large for more than 12 qubits. "
                "Use HM(beta) to build the circuit mixer instead."
            )

        _, Xp, _ = self._paulis()
        Yp = self._pauli_y()

        dim = 2 ** self.num_qubits
        HM = np.zeros((dim, dim), dtype=complex)

        if self.mixer == "x":
            for i in range(self.num_qubits):
                HM += self._pauli_kron(Xp, i)

        elif self.mixer == "xy":
            for q1, q2 in self._xy_mixer_pairs():
                HM += 0.5 * self._two_pauli_kron(Xp, q1, Xp, q2)
                HM += 0.5 * self._two_pauli_kron(Yp, q1, Yp, q2)

        elif self.mixer == "grover":
            uniform = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
            HM = np.outer(uniform, uniform.conj())

        self.HM_matrix = HM
        return HM


    def HC(self, gamma: float) -> QuantumCircuit:
        """
        Cost-phase unitary for the Ising Hamiltonian built by build_HC.
        """
        if not hasattr(self, "h") or not hasattr(self, "J"):
            self.build_HC()

        qc = QuantumCircuit(self.num_qubits, name="U_C")

        for i, h_i in enumerate(self.h):
            if not np.isclose(h_i, 0.0):
                qc.rz(2.0 * gamma * h_i, i)

        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                j_ij = self.J[i, j]
                if np.isclose(j_ij, 0.0):
                    continue

                qc.cx(i, j)
                qc.rz(2.0 * gamma * j_ij, j)
                qc.cx(i, j)

        return qc
    
    
    def HM(self, beta: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name="U_M")

        if self.mixer == "x":
            for q in range(self.num_qubits):
                qc.rx(2.0 * beta, q)

            return qc

        if self.mixer == "xy":
            xy_gate = self._xy_gate(beta)

            for q1, q2 in self._xy_mixer_pairs():
                qc.append(xy_gate, [q1, q2])

            return qc

        qc.compose(self._grover_mixer(beta), inplace=True)

        return qc


    def _xy_gate(self, beta: float) -> UnitaryGate:
        """
        Two-qubit XY mixer gate.

        It implements exp[-i beta/2 (XX + YY)], rotating |01> and |10>.
        This preserves Hamming weight on the selected pair, but unlike the
        Grover mixer it does not preserve all TSP row and column
        constraints by itself.
        """
        U = np.eye(4, dtype=complex)

        c = np.cos(beta)
        s = -1j * np.sin(beta)

        U[1, 1] = c
        U[2, 2] = c
        U[1, 2] = s
        U[2, 1] = s

        return UnitaryGate(U, label="XY")
    

    def _grover_mixer(self, beta: float) -> QuantumCircuit:
        """
        Grover-style mixer over the full computational basis.

        This implements exp(-i beta |s><s|), with |s> the uniform
        superposition. Up to a global phase at beta=pi, this is the usual
        inversion-about-the-mean diffusion step.
        """
        qc = QuantumCircuit(self.num_qubits, name="Grover")

        if self.num_qubits == 0:
            return qc

        for q in range(self.num_qubits):
            qc.h(q)
            qc.x(q)

        if self.num_qubits == 1:
            qc.p(-beta, 0)
        else:
            qc.mcp(-beta, list(range(self.num_qubits - 1)), self.num_qubits - 1)

        for q in range(self.num_qubits):
            qc.x(q)
            qc.h(q)

        return qc

    def _unitary_evolution(self, gamma: float, beta: float) -> QuantumCircuit:
        """
        Build one QAOA layer

            U_M(beta) U_C(gamma)

        using decomposed gate circuits.
        """
        qc = QuantumCircuit(self.num_qubits, name="QAOA_layer")

        qc.compose(self.HC(gamma), inplace=True)
        qc.compose(self.HM(beta), inplace=True)

        return qc


    def _energy(self, theta: np.ndarray) -> float:
        gammas = theta[:self.layers]
        betas = theta[self.layers:]

        qc = self.processor(gammas, betas)
        state = Statevector.from_instruction(qc)
        probabilities = state.probabilities_dict()

        energy = 0.0

        for bitstring, prob in probabilities.items():
            route = self._bitstring_to_one_hot(bitstring)
            energy += prob * self._classical_cost(route)

        return float(np.real(energy))


    def _measure(self, theta: np.ndarray, shots: int = 10):
        """
        Sample bitstrings from the optimized QAOA circuit and return the best route.
        """
        gammas = theta[:self.layers]
        betas = theta[self.layers:]

        qc = self.processor(gammas, betas)
        state = Statevector.from_instruction(qc)

        probabilities = state.probabilities_dict()

        bitstrings = list(probabilities.keys())
        probs = np.array(list(probabilities.values()), dtype=float)
        probs = probs / probs.sum()

        rng = np.random.default_rng(getattr(self, "seed", None))
        samples = rng.choice(bitstrings, size=shots, p=probs)

        best_route = None
        best_reduced_route = None
        best_tour = None
        best_cost = float("inf")
        valid_samples = 0

        for bitstring in samples:
            route = self._bitstring_to_one_hot(bitstring)

            row_valid = np.all(route.sum(axis=1) == 1)
            col_valid = np.all(route.sum(axis=0) == 1)

            if row_valid and col_valid:
                valid_samples += 1
                tour = self._one_hot_to_tour(route)
                cost = self.tsp.tour_cost(tour)

                if cost < best_cost:
                    best_cost = cost
                    best_reduced_route = route
                    best_route = self._expand_one_hot(route)
                    best_tour = tour

        return {
            "best_route": best_route,
            "best_reduced_route": best_reduced_route,
            "best_tour": best_tour,
            "best_cost": best_cost,
            "valid_samples": valid_samples,
            "valid_fraction": valid_samples / shots,
            "samples": samples,
        }


    def _compressed_state(self, theta: np.ndarray) -> np.ndarray:
        if not hasattr(self, "compressed_routes"):
            self.build_compressed_basis()

        gammas = theta[:self.layers]
        betas = theta[self.layers:]

        state = np.zeros(self.compressed_dimension, dtype=complex)
        state[self.compressed_initial_index] = 1.0

        eigvals = self.compressed_mixer_eigvals
        eigvecs = self.compressed_mixer_eigvecs
        eigvecs_dagger = eigvecs.conj().T

        for layer in range(self.layers):
            state *= np.exp(-1j * gammas[layer] * self.compressed_costs)

            mixer_phase = np.exp(-1j * betas[layer] * eigvals)
            state = eigvecs @ (mixer_phase * (eigvecs_dagger @ state))

        return state


    def _compressed_energy(self, theta: np.ndarray) -> float:
        state = self._compressed_state(theta)
        probabilities = np.abs(state) ** 2

        return float(np.real(np.dot(probabilities, self.compressed_costs)))


    def _compressed_measure(self, theta: np.ndarray, shots: int = 10):
        state = self._compressed_state(theta)
        probabilities = np.abs(state) ** 2
        probabilities = probabilities / probabilities.sum()

        rng = np.random.default_rng(getattr(self, "seed", None))
        samples = rng.choice(
            np.arange(self.compressed_dimension),
            size=shots,
            p=probabilities,
        )

        valid_samples = [int(idx) for idx in samples if int(idx) < self.compressed_valid_dimension]

        if valid_samples:
            best_index = min(valid_samples, key=lambda idx: self.compressed_costs[idx])
        else:
            best_index = int(np.argmin(self.compressed_valid_costs))

        best_tour = self.compressed_routes[int(best_index)]
        best_reduced_route = self._tour_to_reduced_one_hot(best_tour)

        return {
            "best_route": self._expand_one_hot(best_reduced_route),
            "best_reduced_route": best_reduced_route,
            "best_tour": best_tour,
            "best_cost": float(self.compressed_costs[int(best_index)]),
            "valid_samples": len(valid_samples),
            "valid_fraction": len(valid_samples) / shots,
            "samples": samples,
            "probabilities": probabilities,
        }


    def _optimize_compressed(self, maxiter: int = 50, shots: int = 10):
        if not hasattr(self, "compressed_routes"):
            self.build_compressed_basis()

        rng = np.random.default_rng(getattr(self, "seed", None))

        initial_theta = np.concatenate([
            rng.uniform(0, 2 * np.pi, self.layers),
            rng.uniform(0, np.pi, self.layers),
        ])

        result = minimize(
            self._compressed_energy,
            initial_theta,
            method="COBYLA",
            options={"maxiter": maxiter},
        )

        theta_opt = result.x
        measurement = self._compressed_measure(theta_opt, shots=shots)

        self.optimal_theta = theta_opt
        self.optimal_gammas = theta_opt[:self.layers]
        self.optimal_betas = theta_opt[self.layers:]
        self.optimization_result = result
        self.best_route = measurement["best_route"]
        self.best_reduced_route = measurement["best_reduced_route"]
        self.best_tour = measurement["best_tour"]
        self.best_cost = measurement["best_cost"]

        return {
            "theta": theta_opt,
            "gammas": self.optimal_gammas,
            "betas": self.optimal_betas,
            "energy": result.fun,
            "best_route": self.best_route,
            "best_reduced_route": self.best_reduced_route,
            "best_tour": self.best_tour,
            "best_cost": self.best_cost,
            "valid_fraction": measurement["valid_fraction"],
            "optimizer_result": result,
            "compressed_basis": True,
            "compressed_dimension": self.compressed_dimension,
            "compressed_valid_dimension": self.compressed_valid_dimension,
            "compressed_qubits": self.compressed_qubits,
            "compressed_unused_states": self.compressed_unused_states,
            "compressed_basis_scope": self.compressed_basis_scope,
            "statevector_dim": self.compressed_dimension,
        }


    def _optimize(self, maxiter: int = 50, shots: int = 10):
        if self.compressed_basis:
            return self._optimize_compressed(maxiter=maxiter, shots=shots)

        if not hasattr(self, "h") or not hasattr(self, "J"):
            self.build_HC()

        rng = np.random.default_rng(getattr(self, "seed", None))

        initial_theta = np.concatenate([
            rng.uniform(0, 2 * np.pi, self.layers),
            rng.uniform(0, np.pi, self.layers),
        ])

        result = minimize(
            self._energy,
            initial_theta,
            method="COBYLA",
            options={"maxiter": maxiter},
        )

        theta_opt = result.x
        measurement = self._measure(theta_opt, shots=shots)

        self.optimal_theta = theta_opt
        self.optimal_gammas = theta_opt[:self.layers]
        self.optimal_betas = theta_opt[self.layers:]
        self.optimization_result = result
        self.best_route = measurement["best_route"]
        self.best_reduced_route = measurement["best_reduced_route"]
        self.best_tour = measurement["best_tour"]
        self.best_cost = measurement["best_cost"]

        return {
            "theta": theta_opt,
            "gammas": self.optimal_gammas,
            "betas": self.optimal_betas,
            "energy": result.fun,
            "best_route": self.best_route,
            "best_reduced_route": self.best_reduced_route,
            "best_tour": self.best_tour,
            "best_cost": self.best_cost,
            "valid_fraction": measurement["valid_fraction"],
            "optimizer_result": result,
        }
    
    def processor(self, gammas, betas):
        qc = self.ansatz()

        for layer in range(self.layers):
            qc.compose(
                self._unitary_evolution(gammas[layer], betas[layer]),
                inplace=True,
            )

        return qc
    
    
def one_hot_encoding(route):
    """
    Convert a TSP tour into an n x n one-hot matrix.

    Example:
        route = [0, 2, 1, 0]

    gives:
        x[0,0] = 1
        x[2,1] = 1
        x[1,2] = 1
    """
    cities = route[:-1]
    n = len(cities)

    x = np.zeros((n, n), dtype=int)

    for t, city in enumerate(cities):
        x[city, t] = 1

    return x

def fixed_start_one_hot_encoding(route, start=0):
    """
    Convert a tour [0, ..., 0] into a reduced one-hot matrix.

    City start at position 0 is removed.
    For n cities, the result has shape (n-1, n-1).
    """
    if route[0] != start or route[-1] != start:
        raise ValueError("Tour must start and end at the fixed start city.")

    cities = route[1:-1]
    n = len(route) - 1

    x = np.zeros((n - 1, n - 1), dtype=int)

    for t, city in enumerate(cities, start=1):
        x[city - 1, t - 1] = 1

    return x


if __name__ == "__main__":
    tsp = TSP(n_locations=6, seed=42)
    tsp.create_map()
    tsp.compute_distance_matrix()

    greedy_tour = tsp.greedy_search(start=0)
    greedy_cost = tsp.tour_cost(greedy_tour)

    greedy_one_hot = fixed_start_one_hot_encoding(greedy_tour, start=0)

    print("Greedy tour:", greedy_tour)
    print("Greedy cost:", round(greedy_cost, 3))
    print("Reduced greedy one-hot:")
    print(greedy_one_hot)

    qaoa = QAOA(
        tsp=tsp,
        layers=1,
        initial_route=greedy_one_hot,
        seed=42,
        fixed_start=True,
        compressed_basis=True,
        neighborhood_depth=1,
    )

    print("\nQAOA setup")
    print("----------")
    print("Number of cities:", qaoa.n)
    print("Number of qubits:", qaoa.num_qubits)
    print("Binary statevector dimension:", 2 ** qaoa.num_qubits)

    result = qaoa._optimize(
        maxiter=20,
        shots=10,
    )
    print("Compressed route-basis dimension:", result["compressed_dimension"])

    print("\nQAOA result")
    print("-----------")
    print("Energy:", round(result["energy"], 3))
    print("Best full tour:", result["best_tour"])
    print("Best cost:", result["best_cost"])
    print("Valid fraction:", result["valid_fraction"])
    print("Best full one-hot:")
    print(result["best_route"])
    print("Best reduced one-hot:")
    print(result["best_reduced_route"])
