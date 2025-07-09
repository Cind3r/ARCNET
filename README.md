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
## Architecture Workflow Diagram
<img src=examples/mermaid-diagram-VERT.svg height=3000/>
