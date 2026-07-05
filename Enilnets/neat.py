"""
NEAT (NeuroEvolution of Augmenting Topologies).

Genomes evolve both connection weights and network topology via mutation
(perturb weights, add connection, add node) and crossover aligned by
historical "innovation" markings, with speciation protecting structural
innovations from being immediately outcompeted. This doesn't reuse
`NeuralNet`, since NEAT genomes are variable-topology graphs rather than a
fixed stack of layers -- a `Genome` is its own minimal feedforward network
representation with its own `forward()`.
"""
from collections import deque
import numpy as np
from .activations import activate
from . import constants


class NodeGene:
    __slots__ = ("id", "type", "activation")

    def __init__(self, node_id, node_type, activation="sigmoid"):
        self.id = node_id
        self.type = node_type  # "input", "bias", "hidden", "output"
        self.activation = activation


class ConnectionGene:
    __slots__ = ("in_node", "out_node", "weight", "enabled", "innovation")

    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation)


class InnovationTracker:
    """Assigns globally-consistent innovation numbers to structural mutations
    (new connections, new nodes) so genomes across the population that
    independently discover the same structural change stay comparable."""

    def __init__(self):
        self._next_node_id = 0
        self._next_innovation = 0
        self._connection_innovations = {}  # (in_node, out_node) -> innovation
        self._node_split_cache = {}        # connection innovation -> new node id

    def reset_node_id_counter(self, start):
        self._next_node_id = start

    def get_connection_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self._connection_innovations:
            self._connection_innovations[key] = self._next_innovation
            self._next_innovation += 1
        return self._connection_innovations[key]

    def get_node_for_split(self, connection_innovation):
        if connection_innovation not in self._node_split_cache:
            self._node_split_cache[connection_innovation] = self._next_node_id
            self._next_node_id += 1
        return self._node_split_cache[connection_innovation]


class Genome:
    """A NEAT genome: a set of node genes and connection genes, evaluable as
    a feedforward network. Connections that would form a cycle are rejected
    at mutation time, so the enabled-connection subgraph is always a DAG."""

    def __init__(self, n_inputs, n_outputs, activation="sigmoid", output_activation=None):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.output_activation = output_activation or activation
        self.nodes = {}
        self.connections = {}
        self.fitness = None
        self.adjusted_fitness = None

        for i in range(n_inputs):
            self.nodes[i] = NodeGene(i, "input")
        self.bias_id = n_inputs
        self.nodes[self.bias_id] = NodeGene(self.bias_id, "bias")
        for j in range(n_outputs):
            nid = n_inputs + 1 + j
            self.nodes[nid] = NodeGene(nid, "output", activation=self.output_activation)

    @property
    def input_ids(self):
        return list(range(self.n_inputs))

    @property
    def output_ids(self):
        return list(range(self.n_inputs + 1, self.n_inputs + 1 + self.n_outputs))

    @classmethod
    def minimal(cls, n_inputs, n_outputs, innovation_tracker, activation="sigmoid",
                output_activation=None, weight_range=1.0):
        """A fully-connected genome with no hidden nodes: every input (plus
        the bias node) connects directly to every output."""
        genome = cls(n_inputs, n_outputs, activation, output_activation)
        for i in genome.input_ids + [genome.bias_id]:
            for j in genome.output_ids:
                innov = innovation_tracker.get_connection_innovation(i, j)
                w = np.random.uniform(-weight_range, weight_range)
                genome.connections[innov] = ConnectionGene(i, j, w, True, innov)
        return genome

    def copy(self):
        g = Genome(self.n_inputs, self.n_outputs, self.activation, self.output_activation)
        g.nodes = {nid: NodeGene(n.id, n.type, n.activation) for nid, n in self.nodes.items()}
        g.connections = {innov: c.copy() for innov, c in self.connections.items()}
        g.fitness = self.fitness
        return g

    def _topo_order(self):
        """Kahn's algorithm over enabled connections. Input/bias nodes always
        start with in-degree 0; nodes unreachable via any enabled connection
        also get in-degree 0 and are evaluated as a zero-input node."""
        all_nodes = list(self.nodes.keys())
        in_degree = {nid: 0 for nid in all_nodes}
        out_edges = {nid: [] for nid in all_nodes}
        for conn in self.connections.values():
            if not conn.enabled:
                continue
            out_edges[conn.in_node].append(conn.out_node)
            in_degree[conn.out_node] += 1

        queue = deque(nid for nid in all_nodes if in_degree[nid] == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for tgt in out_edges[nid]:
                in_degree[tgt] -= 1
                if in_degree[tgt] == 0:
                    queue.append(tgt)
        return order

    def break_cycles(self):
        """Disable the minimum number of enabled connections needed to make
        the enabled-connection subgraph acyclic. Each individual mutation
        (add_connection, add_node) already only ever extends an acyclic
        graph, but crossover's matching-gene coin flip can independently pick
        each endpoint of a same-innovation edge from a different parent --
        and since the two parents are only guaranteed acyclic *individually*,
        not in combination, that can reintroduce a cycle the disabled copy
        was specifically there to prevent. This is the safety net."""
        while True:
            order = self._topo_order()
            if len(order) == len(self.nodes):
                return
            stuck = set(self.nodes.keys()) - set(order)
            for conn in self.connections.values():
                if conn.enabled and conn.in_node in stuck and conn.out_node in stuck:
                    conn.enabled = False
                    break
            else:
                return  # no enabled edge found among stuck nodes; nothing more to do

    def forward(self, x):
        """x: (n_inputs,) or (batch, n_inputs). Returns matching-rank output
        of shape (n_outputs,) or (batch, n_outputs)."""
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        batch = x.shape[0]

        incoming = {}
        for conn in self.connections.values():
            if conn.enabled:
                incoming.setdefault(conn.out_node, []).append((conn.in_node, conn.weight))

        values = {i: x[:, i] for i in range(self.n_inputs)}
        values[self.bias_id] = np.ones(batch, dtype=np.float64)

        for nid in self._topo_order():
            node = self.nodes[nid]
            if node.type in ("input", "bias"):
                continue
            srcs = incoming.get(nid)
            s = np.zeros(batch, dtype=np.float64) if not srcs else \
                sum(values[src] * w for src, w in srcs)
            values[nid] = activate(node.activation, s)

        out = np.stack([values[nid] for nid in self.output_ids], axis=1)
        return out[0] if single else out

    def mutate_weights(self, rate=None, perturb_rate=None, perturb_power=None, replace_range=1.0):
        rate = constants.NEAT_WEIGHT_MUTATE_RATE if rate is None else rate
        perturb_rate = constants.NEAT_WEIGHT_PERTURB_RATE if perturb_rate is None else perturb_rate
        perturb_power = constants.NEAT_WEIGHT_PERTURB_POWER if perturb_power is None else perturb_power
        for conn in self.connections.values():
            if np.random.rand() >= rate:
                continue
            if np.random.rand() < perturb_rate:
                conn.weight += np.random.randn() * perturb_power
            else:
                conn.weight = np.random.uniform(-replace_range, replace_range)

    def _creates_cycle(self, in_node, out_node):
        if in_node == out_node:
            return True
        adj = {}
        for conn in self.connections.values():
            if conn.enabled:
                adj.setdefault(conn.in_node, []).append(conn.out_node)
        stack, visited = [out_node], set()
        while stack:
            n = stack.pop()
            if n == in_node:
                return True
            if n in visited:
                continue
            visited.add(n)
            stack.extend(adj.get(n, []))
        return False

    def mutate_add_connection(self, innovation_tracker, max_tries=20, weight_range=1.0):
        """Add a new enabled connection between two previously-unconnected
        nodes, rejecting attempts that would create a cycle or feed into an
        input/bias node. Returns True if a connection was added."""
        node_ids = list(self.nodes.keys())
        existing = {(c.in_node, c.out_node) for c in self.connections.values()}
        for _ in range(max_tries):
            a = node_ids[np.random.randint(len(node_ids))]
            b = node_ids[np.random.randint(len(node_ids))]
            if self.nodes[b].type in ("input", "bias"):
                continue
            if (a, b) in existing or self._creates_cycle(a, b):
                continue
            innov = innovation_tracker.get_connection_innovation(a, b)
            w = np.random.uniform(-weight_range, weight_range)
            self.connections[innov] = ConnectionGene(a, b, w, True, innov)
            return True
        return False

    def mutate_add_node(self, innovation_tracker):
        """Split a random enabled connection in two via a new hidden node:
        disable the original, add in->new (weight 1) and new->out (the
        original weight), preserving the connection's function at the moment
        of the split. Returns True if a node was added."""
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return False
        conn = enabled[np.random.randint(len(enabled))]
        conn.enabled = False

        new_id = innovation_tracker.get_node_for_split(conn.innovation)
        self.nodes[new_id] = NodeGene(new_id, "hidden", activation=self.activation)

        innov1 = innovation_tracker.get_connection_innovation(conn.in_node, new_id)
        innov2 = innovation_tracker.get_connection_innovation(new_id, conn.out_node)
        self.connections[innov1] = ConnectionGene(conn.in_node, new_id, 1.0, True, innov1)
        self.connections[innov2] = ConnectionGene(new_id, conn.out_node, conn.weight, True, innov2)
        return True

    def distance(self, other, c1=None, c2=None, c3=None):
        """Compatibility distance for speciation: weighted count of excess +
        disjoint genes (normalized by genome size) plus average weight
        difference on matching genes."""
        c1 = constants.NEAT_C1 if c1 is None else c1
        c2 = constants.NEAT_C2 if c2 is None else c2
        c3 = constants.NEAT_C3 if c3 is None else c3

        innov_a = set(self.connections.keys())
        innov_b = set(other.connections.keys())
        matching = innov_a & innov_b
        only_a, only_b = innov_a - innov_b, innov_b - innov_a
        lower_max = min(max(innov_a, default=0), max(innov_b, default=0))
        disjoint = sum(1 for i in only_a if i <= lower_max) + sum(1 for i in only_b if i <= lower_max)
        excess = sum(1 for i in only_a if i > lower_max) + sum(1 for i in only_b if i > lower_max)

        if matching:
            weight_diff = float(np.mean([abs(self.connections[i].weight - other.connections[i].weight)
                                         for i in matching]))
        else:
            weight_diff = 0.0

        n = max(len(innov_a), len(innov_b))
        n = 1 if n < 20 else n
        return (c1 * excess + c2 * disjoint) / n + c3 * weight_diff


def crossover(fitter, other, disjoint_disable_rate=0.75):
    """Combine two genomes into a child, assuming fitter.fitness >=
    other.fitness. Matching genes (by innovation number) are inherited
    randomly from either parent; disjoint/excess genes come only from the
    fitter parent (standard NEAT rule -- the less fit parent's unique
    structure is dropped)."""
    child = Genome(fitter.n_inputs, fitter.n_outputs, fitter.activation, fitter.output_activation)
    for innov, conn_a in fitter.connections.items():
        conn_b = other.connections.get(innov)
        new_conn = (conn_b if conn_b is not None and np.random.rand() < 0.5 else conn_a).copy()
        if (not conn_a.enabled or (conn_b is not None and not conn_b.enabled)) and \
                np.random.rand() < disjoint_disable_rate:
            new_conn.enabled = False
        child.connections[innov] = new_conn
        for nid in (new_conn.in_node, new_conn.out_node):
            if nid not in child.nodes:
                src = fitter.nodes.get(nid) or other.nodes.get(nid)
                child.nodes[nid] = NodeGene(nid, src.type, src.activation)
    child.break_cycles()
    return child


class Species:
    def __init__(self, representative, species_id):
        self.id = species_id
        self.representative = representative
        self.members = []
        self.best_fitness = -np.inf
        self.staleness = 0

    def update_stagnation(self, best_fitness):
        if best_fitness > self.best_fitness + 1e-9:
            self.best_fitness = best_fitness
            self.staleness = 0
        else:
            self.staleness += 1


class NEATPopulation:
    """Evolves a population of `Genome`s against a user-supplied fitness
    function via mutation, speciation, and fitness-shared reproduction.

    Example
    -------
    >>> pop = NEATPopulation(n_inputs=2, n_outputs=1, population_size=150)
    >>> def fitness(genome):
    ...     preds = np.array([genome.forward(x)[0] for x in xor_inputs])
    ...     return float(4 - np.sum((preds - xor_targets) ** 2))
    >>> history = pop.evolve(fitness, generations=100)
    >>> best = pop.best_genome
    >>> best.forward(xor_inputs[0])
    """

    def __init__(self, n_inputs, n_outputs, population_size=150, activation="sigmoid",
                 output_activation=None,
                 compatibility_threshold=None, c1=None, c2=None, c3=None,
                 weight_mutate_rate=None, weight_perturb_rate=None, weight_perturb_power=None,
                 add_connection_rate=None, add_node_rate=None, crossover_rate=None,
                 survival_threshold=None, stagnation_limit=None, elitism=1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.population_size = population_size
        self.activation = activation
        self.output_activation = output_activation or activation

        self.compatibility_threshold = constants.NEAT_COMPATIBILITY_THRESHOLD \
            if compatibility_threshold is None else compatibility_threshold
        self.c1 = constants.NEAT_C1 if c1 is None else c1
        self.c2 = constants.NEAT_C2 if c2 is None else c2
        self.c3 = constants.NEAT_C3 if c3 is None else c3
        self.weight_mutate_rate = constants.NEAT_WEIGHT_MUTATE_RATE if weight_mutate_rate is None else weight_mutate_rate
        self.weight_perturb_rate = constants.NEAT_WEIGHT_PERTURB_RATE if weight_perturb_rate is None else weight_perturb_rate
        self.weight_perturb_power = constants.NEAT_WEIGHT_PERTURB_POWER if weight_perturb_power is None else weight_perturb_power
        self.add_connection_rate = constants.NEAT_ADD_CONNECTION_RATE if add_connection_rate is None else add_connection_rate
        self.add_node_rate = constants.NEAT_ADD_NODE_RATE if add_node_rate is None else add_node_rate
        self.crossover_rate = constants.NEAT_CROSSOVER_RATE if crossover_rate is None else crossover_rate
        self.survival_threshold = constants.NEAT_SURVIVAL_THRESHOLD if survival_threshold is None else survival_threshold
        self.stagnation_limit = constants.NEAT_STAGNATION_LIMIT if stagnation_limit is None else stagnation_limit
        self.elitism = elitism

        self.innovation_tracker = InnovationTracker()
        self.innovation_tracker.reset_node_id_counter(n_inputs + 1 + n_outputs)
        self.genomes = [
            Genome.minimal(n_inputs, n_outputs, self.innovation_tracker,
                          activation=activation, output_activation=self.output_activation)
            for _ in range(population_size)
        ]
        self.species = []
        self._next_species_id = 0
        self.generation = 0
        self.best_genome = None
        self.best_fitness = -np.inf

    def _speciate(self):
        for sp in self.species:
            sp.members = []
        for genome in self.genomes:
            placed = False
            for sp in self.species:
                if genome.distance(sp.representative, self.c1, self.c2, self.c3) < self.compatibility_threshold:
                    sp.members.append(genome)
                    placed = True
                    break
            if not placed:
                sp = Species(genome.copy(), self._next_species_id)
                self._next_species_id += 1
                sp.members.append(genome)
                self.species.append(sp)
        self.species = [sp for sp in self.species if sp.members]
        for sp in self.species:
            sp.representative = sp.members[np.random.randint(len(sp.members))].copy()

    def evolve(self, fitness_fn, generations=100, verbose=True):
        """fitness_fn(genome) -> float, higher is better. Returns a list of
        the best-fitness-seen-so-far, one entry per generation."""
        history = []
        for gen in range(generations):
            for genome in self.genomes:
                genome.fitness = float(fitness_fn(genome))

            gen_best = max(self.genomes, key=lambda g: g.fitness)
            if gen_best.fitness > self.best_fitness:
                self.best_fitness = gen_best.fitness
                self.best_genome = gen_best.copy()
            history.append(self.best_fitness)
            if verbose:
                print(f"Generation {gen + 1}/{generations} - best fitness: {self.best_fitness:.4f} "
                     f"(gen best: {gen_best.fitness:.4f}, species: {max(len(self.species), 1)})")

            self._speciate()
            for sp in self.species:
                sp.update_stagnation(max(m.fitness for m in sp.members))

            active_species = [sp for sp in self.species if sp.staleness < self.stagnation_limit] or self.species
            for sp in active_species:
                for m in sp.members:
                    m.adjusted_fitness = m.fitness / len(sp.members)
            total_adjusted = sum(m.adjusted_fitness for sp in active_species for m in sp.members)
            if total_adjusted <= 0:
                total_adjusted = 1.0

            new_genomes = []
            if self.elitism > 0:
                for g in sorted(self.genomes, key=lambda g: g.fitness, reverse=True)[:self.elitism]:
                    new_genomes.append(g.copy())

            for sp in active_species:
                sp_adjusted_sum = sum(m.adjusted_fitness for m in sp.members)
                n_offspring = int(round(self.population_size * sp_adjusted_sum / total_adjusted))
                ranked = sorted(sp.members, key=lambda g: g.fitness, reverse=True)
                n_parents = max(1, int(np.ceil(len(ranked) * self.survival_threshold)))
                parents = ranked[:n_parents]
                for _ in range(n_offspring):
                    if len(parents) > 1 and np.random.rand() < self.crossover_rate:
                        pa, pb = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
                        if pa.fitness < pb.fitness:
                            pa, pb = pb, pa
                        child = crossover(pa, pb)
                    else:
                        child = parents[np.random.randint(len(parents))].copy()
                    if np.random.rand() < self.add_connection_rate:
                        child.mutate_add_connection(self.innovation_tracker)
                    if np.random.rand() < self.add_node_rate:
                        child.mutate_add_node(self.innovation_tracker)
                    child.mutate_weights(self.weight_mutate_rate, self.weight_perturb_rate, self.weight_perturb_power)
                    new_genomes.append(child)

            while len(new_genomes) < self.population_size:
                parent = self.genomes[np.random.randint(len(self.genomes))]
                child = parent.copy()
                child.mutate_weights(self.weight_mutate_rate, self.weight_perturb_rate, self.weight_perturb_power)
                new_genomes.append(child)

            self.genomes = new_genomes[:self.population_size]
            self.generation += 1

        return history
