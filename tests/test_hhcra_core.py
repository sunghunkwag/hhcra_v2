import unittest

import numpy as np

from hhcra_v2 import (
    CausalGNN,
    CausalGraphData,
    HHCRAConfig,
    LiquidNeuralNet,
    NeuroSymbolicEngine,
)


class HHCRACoreRegressionTests(unittest.TestCase):
    def small_config(self) -> HHCRAConfig:
        return HHCRAConfig(
            obs_dim=6,
            latent_dim=3,
            num_true_vars=3,
            num_vars=4,
            edge_threshold=0.5,
            liquid_ode_steps=2,
            liquid_dt=0.05,
            train_epochs_l1=1,
            train_epochs_l2=1,
            train_epochs_l3=1,
        )

    def test_causal_graph_traversal_and_dag_check(self):
        adjacency = np.zeros((4, 4))
        edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 3, 1.0),
        ]
        for parent, child, _ in edges:
            adjacency[child, parent] = 1.0

        graph = CausalGraphData(nodes=[0, 1, 2, 3], edges=edges, adjacency=adjacency)

        self.assertTrue(graph.is_dag())
        self.assertEqual(graph.parents(2), {1})
        self.assertEqual(graph.children(0), {1, 3})
        self.assertEqual(graph.ancestors(2), {0, 1})
        self.assertEqual(graph.descendants(0), {1, 2, 3})
        self.assertTrue(graph.has_edge(1, 2))
        self.assertEqual(graph.edge_count(), 3)

    def test_d_separation_blocks_observed_chain_middle(self):
        adjacency = np.zeros((3, 3))
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        for parent, child, _ in edges:
            adjacency[child, parent] = 1.0

        graph = CausalGraphData(nodes=[0, 1, 2], edges=edges, adjacency=adjacency)
        engine = NeuroSymbolicEngine()

        self.assertFalse(engine.d_separated(graph, 0, 2, set()))
        self.assertTrue(engine.d_separated(graph, 0, 2, {1}))

    def test_gnn_prune_to_dag_removes_cycles_and_self_loops(self):
        config = self.small_config()
        gnn = CausalGNN(config)

        # W[child, parent] > threshold means parent -> child.
        gnn.W[:] = -5.0
        np.fill_diagonal(gnn.W, -10.0)
        gnn.W[1, 0] = 5.0  # 0 -> 1
        gnn.W[2, 1] = 4.0  # 1 -> 2
        gnn.W[0, 2] = 3.0  # 2 -> 0, creates a directed cycle

        self.assertFalse(gnn._is_dag(gnn.adjacency(hard=True)))
        gnn.prune_to_dag()
        pruned = gnn.adjacency(hard=True)

        self.assertTrue(gnn._is_dag(pruned))
        self.assertTrue(np.all(np.diag(pruned) == 0.0))

    def test_liquid_net_evolve_and_intervene_are_finite_and_shape_safe(self):
        config = self.small_config()
        liquid = LiquidNeuralNet(config)
        embeddings = np.ones((2, 3, config.num_vars, config.latent_dim), dtype=float) * 0.25
        adjacency = np.zeros((config.num_vars, config.num_vars), dtype=float)
        adjacency[1, 0] = 1.0
        adjacency[2, 1] = 1.0

        evolved = liquid.evolve(embeddings, adjacency)
        intervened = liquid.intervene(
            embeddings,
            adjacency,
            {1: np.full(config.latent_dim, 0.75, dtype=float)},
        )

        self.assertEqual(evolved.shape, embeddings.shape)
        self.assertEqual(intervened.shape, embeddings.shape)
        self.assertTrue(np.all(np.isfinite(evolved)))
        self.assertTrue(np.all(np.isfinite(intervened)))


if __name__ == "__main__":
    unittest.main()
