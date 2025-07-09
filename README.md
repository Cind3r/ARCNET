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

To avoid scrolling, `ARCNET.core.trainer` was broken down into horizontal sections meant to be read L -> R. Any arrow coming off a subgraph (e.g., "Initialization") is coming from the RIGHT MOST block in the workflow diagram. This is most confusing for 'Lineage & Pruning' where the arrows "Yes" and "No" are both coming from "All Steps Done?".

```mermaid
---
config:
  layout: elk
---
flowchart TD
 subgraph Init["Initialization"]
    direction LR
        B{"Enable Model Save?"}
        A["Start: run_aan_comprehensive_q_learning<br>(core/trainer.py)"]
        E["Initialize Lineage Registry"]
        C["load_and_seed_population_fixed<br>(data/loader.py)"]
        D["Create ConceptModule(s)<br>(models/arcnet_learner.py)"]
  end
 subgraph Data["Data Prep"]
    direction LR
        G["For each Evolution Step (steps)"]
        F["Convert Data to Tensors<br>(core/trainer.py)"]
  end
 subgraph Train["Training Loop"]
    direction LR
        H2{"training_method"}
        H1["For each Module in Population"]
        L["Q-Learning Update<br>(m.update_q, core/QModule.py)"]
        J["Set m.fitness"]
        I1["compute_fitness_adaptive_complexity_enhanced<br>(evolution/fitness.py)"]
        K["Set m.loss, m.fitness = -m.loss"]
        I2["Gradient Descent (compute_loss + optimizer)<br>(evolution/loss.py)"]
  end
 subgraph QBias["Q-Learning & Bias Detection"]
    direction LR
        O{"Enable Bias Elimination?"}
        N["Bias Detection<br>(monitor_prediction_diversity_with_action, evolution/bias.py)"]
        M["Message Passing<br>(comprehensive_manifold_q_message_passing, core/message.py)"]
        Q["Continue"]
        P["catalyst_bias_elimination<br>(evolution/bias.py)"]
  end
 subgraph Track["Model Tracking"]
    direction LR
        R{"track_best_models?"}
        U["Q-Guided Survival Selection<br>(core/trainer.py)"]
        S["Track Best Models/Stats<br>(core/trainer.py)"]
        T["Skip Tracking"]
  end
 subgraph Repro["Reproduction & Mutation"]
    direction LR
        X["Population Update"]
        W["Q-Experience Inheritance<br>(core/QModule.py)"]
        V["Q-Guided Reproduction &amp; Mutation<br>(mutate, choose_action, models/arcnet_learner.py)"]
  end
 subgraph Lineage["Lineage & Pruning"]
    direction LR
        Y{"enable_lineage_snap?"}
        AB{"Lineage Pruning Needed?"}
        Z["save_lineage_snapshot_to_file<br>(data/loader.py)"]
        AA["Skip Snapshot"]
        AE["Next Step or Finish"]
        AC["Prune Lineage Registry<br>(core/trainer.py)"]
        AD["Continue"]
        AF{"All Steps Done?"}
  end
 subgraph Final["Finalization"]
    direction LR
        AH{"enable_model_save?"}
        AG["Theorem Checks &amp; Final Reporting<br>(core/trainer.py)"]
        AJ["Export (END)<br>- Final population of models<br>- Best model overall"]
        AI["export_best_models<br>(data/loader.py)"]
  end
    A --> B
    B -- Yes, Resume --> C
    C --> E
    B -- No, Fresh --> D
    D --> E
    Init --> Data
    F --> G
    Data --> Train
    H1 --> H2
    H2 -- fitness --> I1
    I1 --> J
    J --> L
    H2 -- loss --> I2
    I2 --> K
    K --> L
    Train --> QBias
    M --> N
    N --> O
    O -- Yes, Needed --> P
    P --> Q
    O -- No/Not Needed --> Q
    QBias --> Track
    R -- Yes --> S
    S --> U
    R -- No --> T
    T --> U
    Track --> Repro
    V --> W
    W --> X
    Repro --> Lineage
    Y -- Yes --> Z
    Z --> AB
    Y -- No --> AA
    AA --> AB
    AB -- Yes --> AC
    AC --> AE
    AB -- No --> AD
    AD --> AE
    AE --> AF
    Lineage -- Yes --> Final
    Lineage -- No --> Data
    AG --> AH
    AH -- Yes --> AI
    AI --> AJ
    AH -- No --> AJ
     A:::main
     B:::main
     C:::main
     E:::main
     D:::main
     F:::main
     G:::main
     H1:::main
     H2:::main
     I1:::main
     I1:::op
     J:::main
     L:::main
     L:::qlearn
     I2:::main
     I2:::op
     K:::main
     M:::main
     N:::main
     N:::bias
     O:::main
     P:::main
     P:::bias
     Q:::main
     R:::main
     S:::main
     S:::track
     U:::main
     U:::track
     T:::main
     V:::main
     V:::repro
     W:::main
     W:::repro
     X:::main
     X:::repro
     Y:::main
     Z:::main
     AB:::main
     AA:::main
     AC:::main
     AC:::prune
     AE:::main
     AD:::main
     AF:::main
     AG:::main
     AG:::theorem
     AH:::main
     AI:::main
     AI:::theorem
     AJ:::main
     AJ:::theorem
    classDef main fill:#333333,stroke:#00796b,stroke-width:3px,color:#FFFFFF
    classDef branch fill:#333333,stroke:#fbc02d,stroke-width:3px
    classDef op fill:#333333,stroke:#f57c00,stroke-width:3px
    classDef qlearn fill:#333333,stroke:#7b1fa2,stroke-width:3px
    classDef bias fill:#333333,stroke:#c62828,stroke-width:3px
    classDef track fill:#333333,stroke:#388e3c,stroke-width:3px
    classDef repro fill:#333333,stroke:#1976d2,stroke-width:3px
    classDef pop fill:#333333,stroke:#827717,stroke-width:3px
    classDef prune fill:#333333,stroke:#5d4037,stroke-width:3px
    classDef theorem fill:#333333,stroke:#ad1457,stroke-width:3px

```

