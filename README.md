# ARCNET
Autocatalytic, Relativistic, Curvature, **Net**work

## Autocatalytic Abstraction Networks (AANs)


Conventional ML systems are static learners: fixed objectives, fixed architectures, fixed inductive biases. Even AutoML, meta-learning, or neural architecture search operate within sandboxed limits. AANs instead:

Continuously evolve their own internal abstractions, representations, and architectural modules through a process similar to autocatalytic chemical networks in origin-of-life theories.

#### On Self-Organizing Architectures
While not explicitly autocatalytic, modern ML models learn hierarchies of abstractions:

- Convolutional Neural Networks (CNNs) learn from pixels -> edges -> textures -> objects.

- Transformers in NLP or vision (e.g., GPT, BERT, ViT) learn compositional, context-aware representations that feed into deeper ones.

- These representations bootstrap one another during training, a primitive form of abstraction catalysis.

However, these networks require external data and supervision; they are not autocatalytic in the self-sustaining sense.

**DreamCoder** in particular comes very close:
- It discovers reusable abstractions (like functions or patterns),

- Learns how to use them to solve new problems,

- And improves its own language of thought over time.

This is an explicit example of an abstraction network with *partial* autocatalytic features. To fully qualify as an autocatalytic abstraction network, a system would need to:

1. Generate its own abstractions
2. Use those to create further abstractions
3. Do this recursively and indefinitely
4. Without heavy external steering

We're going to see if we can improve the process of higher-order abstractions and maintaining accuracy by incorporating manifold folding of abstractions in a 3d space. 

![til](examples/lineage_evolution.gif)
![til](examples/lineage_evolution_3d_depth.gif)

***
## What if we add QISRL (Learning in Curved Decision Spaces)

Curved State-Time Embedding
Rather than modeling the environment as a flat vectorized state, we embed it into a curved latent manifold whose local geometry is conditioned on:

- Information gain
- Reward potential density
- Causality distance (measured via temporal counterfactuals)

### Dynamic Temporal Relativity
We introduce a non-uniform "flow of time" in the RL policy, i.e., the policy updates itself more rapidly in high-entropy, high-surprise regions and slows down in stable ones.
A time-weighted Bellman equation is derived:

$$
Q(s,a) = r(s,a) + \gamma(s) \max\limits_{a'} Q(s',a')
$$

where $\gamma(s) = e^{-\lambda H(s)}$ and $H(s)$ is the entropy or information uncertainty at state $s$.

Inspired by general relativity, we define a metric tensor over the state space that warps distances:

$$
ds^2 = g_{ij}(x) dx^i dx^j
$$

where $g_{ij}(x)$ is learned via a neural net based on reqard curvature and transition predictability.

Instead of choosing a discrete action, the agent holds a superposition of policy branches weighted by complex-valued amplitudes, which collapse during high-reward observations. It’s not quantum computing only inspired.

***
## Architecture Workflow Diagram (ASCII)

+-----------------------------------------------------------------------------------+
|                run Trainer (core/trainer.py)                                      |
+-----------------------------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| Initialization    |  core/trainer.py                                              |
|                   |  - Creates initial population of ConceptModule                |
|                   |  - Initializes lineage, reward history                        |
|                   |  - Uses: models/arcnet.py (ConceptModule)                     |
|                   |         core/blueprint.py (ArchitectureBlueprint, if used)    |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| Main Evolutionary Loop (for each step)                                            |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 1. Module Training/Evaluation                                                     |
|   - For each module in population:                                                |
|     - If 'loss':                                                                  |
|         * Local gradient descent (Adam)                                           |
|           (core/trainer.py, evolution/loss.py: compute_loss)                      |
|         * Set m.fitness = -m.loss                                                 |
|     - If 'fitness':                                                               |
|         * Compute fitness (evolution/fitness.py)                                  |
|     - Model: models/arcnet.py (ConceptModule)                                     |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 2. Q-Learning Update                                                              |
|   - For each module:                                                              |
|     - Compute reward (evolution/rewards.py: compute_reward_adaptive_aan_normalized)|
|     - Update Q-function (core/QModule.py: CompressedQModule)                      |
|     - Q-function is a neural net per module (models/arcnet_learner.py)            |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 3. Message Passing                                                                |
|   - Information flow between modules                                              |
|   - core/message.py: comprehensive_manifold_q_message_passing                     |
|   - models/arcnet_learner.py: process_messages, receive_message                   |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 4. Bias Detection & Elimination                                                   |
|   - Monitor and eliminate biased modules                                          |
|   - evolution/bias.py: monitor_prediction_diversity_with_action,                  |
|                       catalyst_bias_elimination, select_bias_resistant_catalysts  |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 5. Tracking & Monitoring                                                          |
|   - Save best models, track stats, snapshots                                      |
|   - core/trainer.py (best_models_history, generation_stats, save_lineage_snapshot)|
|   - data/loader.py: save_lineage_snapshot_to_file, export_best_models             |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 6. Survival Selection                                                             |
|   - Q-learning influenced selection                                               |
|   - core/trainer.py (sort by m._survival_score)                                   |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 7. Reproduction & Mutation                                                        |
|   - Q-guided target selection, catalyst selection                                 |
|   - models/arcnet.py: mutate, record_assembly_operation                           |
|   - evolution/bias.py: select_bias_resistant_catalysts                            |
|   - Manifold-aware position update (geodesic_interpolate)                         |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 8. Q-Experience Inheritance                                                       |
|   - Offspring inherit Q-experiences from catalysts                                |
|   - core/QModule.py: replay_buffer                                                |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 9. Population Update                                                              |
|   - Survivors + offspring become new population                                   |
|   - core/trainer.py                                                               |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 10. Lineage Pruning                                                               |
|   - Prune lineage registry to keep it manageable                                  |
|   - core/trainer.py                                                               |
+-------------------+---------------------------------------------------------------+
        |
        v
+-------------------+---------------------------------------------------------------+
| 11. Final Reporting                                                               |
|   - Print stats, export models, return results                                    |
|   - core/trainer.py                                                               |
+-------------------+---------------------------------------------------------------+

***
<img src=examples/mermaid-diagram-VERT.svg />
