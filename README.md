[**Main scripts**](#main_scripts) | [**Typical train and test flow**](#train_test_flow) |  [**Citation**](#citation)

<h2> Results </h2>

<h4>Trained with 3 agent and executed with 10 agents</h4>
<img src="https://github.com/jselvaraaj/JaxInforMARL/blob/main/10_agents.gif?raw=true" alt="10 agents" width="60%">

<h4>Trained with 3 agent and executed with 100 agents</h4>
<img src="https://github.com/jselvaraaj/JaxInforMARL/blob/main/100_agents.gif?raw=true" alt="100 agents" width="60%">

<h2 name="main_scripts" id="main_scripts">Main scripts </h2>

1. `algorithm/marl_ppo.py` for training Multi agent PPO on target MPE environment.
    - Note run this script as python module with `python -m algorithm/marl_ppo.py` for imports to work properly.
2. `envs/target_mpe_env.py`. This is the main class that defines the target MPE environment.
    - Also look at `envs/wrapper.py` for env wrappers.
3. `config/mappo_config.py`. This is the one and only file for changing config values to run experiments.
   Used python classes instead of yaml file to get auto complete and type checking and easier refactor when accessing
   and changing the structure of config.
4. `visualize_actor.py` for visualizing the trained actor in a local environment.
5. `model/actor_critic_rnn.py` has all the flax linen networks used in the PPO.

<h2 name="train_test_flow" id="train_test_flow">Typical train and test flow</h2>

1. Run the `train_with_gpu.ipynb` notebook in a colab with gpu.
    - Remember to set up the config in `WandbConfig` in `config/mappo_config.py` and change mode `online` to get wandb
      logging.
    - The artifacts are saved under the name "PPO_RNN_Runner_State"
2. Visualize the actor with `visualize_actor.py` after changing the `artifact_version` variable in the block.
   `if __name__ == "__main__"`

# Note

It is recommended to first install either `requirements_jax_cpu.txt` or `requirements_jax_cuda.txt` before
`requirements.txt` since the packages in `requirements` will install a jax version for you.

<h2 name="citation" id="citation">Citing JaxInforMARL</h2>

If you use JaxInforMARL in your work, please cite as follows:

```
@software{JaxInforMARL,
      title={JaxInforMARL: Multi-Agent Target MPE RL Environments with GNNs in JAX},
      author={Joseph Selvaraaj},
      year = {2025},
      url = {https://github.com/jselvaraaj/JaxInforMARL},
      version = {1.0.0}
    }
```

# Run with WSL
```bash
nvidia-smi

source /home/weaver/.virtualenvs/jaxinformarl/bin/activate
unset LD_LIBRARY_PATH       # Make this not load CUDA-12.9 pip libraries, just cuda-12.5
python -c "import jax; print(jax.devices())"
python ./algorithm/marl_ppo.py  # Training
python visualize_actor.py       # Testing
```

# Legend of `visualize_actor.py`

| Agent | Color | Hex |
|---|---|---|
| `agent_0` (`A 0`) | 🔵 Blue | `#1f77b4` |
| `agent_1` (`A 1`) | 🟠 Orange | `#ff7f0e` |
| `agent_2` (`A 2`) | 🟢 Green | `#2ca02c` |

Each corresponding target (`T 0`–`T 2`) uses the same color as its agent.

## Action mapping

| Action | Movement | Control vector |
|---:|---|---|
| `0` | Stay still | `[0, 0]` |
| `1` | Left | `[-1, 0]` |
| `2` | Right | `[1, 0]` |
| `3` | Down | `[0, -1]` |
| `4` | Up | `[0, 1]` |


The actual control vector is multiplied by the configured `entity_acceleration`.

## Environment-step debug output

The debug block in `algorithm/marl_ppo.py` reports environment `0` from the
vectorized environment batch. Each block describes one transition from the
state before the selected action to the state returned by the environment.

| Debug section | Meaning |
|---|---|
| `Actual actor observations (o(i) + agent node features)` | The six-value environment observation followed by the learned graph embedding passed to `ActorRNN` for action selection. |
| `o(i)` | `[global position, global velocity, goal position relative to the agent]`. |
| `Raw graph node features` | The latest unembedded node features from the viewpoint of each observing agent. Each entity shows its equivariant features followed by its non-equivariant entity type. |
| `Actions` | The policy output for each agent, displayed as action ID, movement name, and control vector. |
| `Rewards` | Rewards returned by the environment after applying the displayed actions. |
| `Global positions before step` | Agent positions belonging to the state used to select the actions. |
| `Global positions after step` | Agent positions returned after the environment applies the actions and advances its dynamics. |

Each raw equivariant graph feature is a `3 x 2` array whose rows are:

| Row | Feature |
|---:|---|
| `0` | Entity position minus the observing agent's position |
| `1` | Entity velocity minus the observing agent's velocity |
| `2` | That entity's assigned goal position minus the observing agent's position |

The final value printed after `|` is the non-equivariant entity type. With
graph attention enabled, the actor builds its learned embedding from the
equivariant node features and graph connectivity. The raw flat environment
observation is placed before this graph-derived embedding when the input to
`ActorRNN` is constructed.

## Current implementation and original InforMARL comparison

Training with `python algorithm/marl_ppo.py` and testing with
`python visualize_actor.py` use the same `_env_step`, observation and graph
construction, environment dynamics, reward function, and actor/critic classes.
The shared path is [`_env_step`](algorithm/marl_ppo.py#L372), which both the
[training loop](algorithm/marl_ppo.py#L871) and
[`visualize_actor.py`](visualize_actor.py#L221) call. The one action-selection
difference is:

```python
action = pi.mode() if is_running_in_viz_mode else pi.sample(seed=_rng)
```

Training therefore samples from the categorical policy, while visualization
uses its deterministic mode. Visualization also reconstructs the environment
with `with_paper_target_env(..., testing=True)`, so it can use different numbers
of agents and episode lengths even though the implementation is shared.

### Observation and graph input

The current local observation is defined in
[`TargetMPEEnvironment.get_observation`](envs/target_mpe_env.py#L332):

```python
return jnp.concatenate(
    [agent_position.flatten(),
     agent_velocity.flatten(),
     landmark_relative_position.flatten()]
)
```

Thus, for agent `i`,
`o(i) = [p(i), v(i), p_goal(i) - p(i)]`. Relative graph-node features are
constructed in [`get_graph`](envs/target_mpe_env.py#L361) as
`[relative_position, relative_velocity, relative_goal]`. After graph attention,
the actor selects agent `i`'s node representation and uses the corrected order:

```python
obs = jnp.concatenate([obs, agent_node_features], axis=-1)
hidden, pi = ActorRNN(self.action_dim, self.config)(hidden, (obs, dones))
```

See [`GraphAttentionActorRNN`](model/actor_critic_rnn.py#L320).

The original paper implementation defines the ordinary observation in
[`navigation_graph.py`](InforMARL/multiagent/custom_scenarios/navigation_graph.py#L399)
as:

```python
return np.concatenate([agent.state.p_vel, agent.state.p_pos] + goal_pos)
```

Its content is the same but its order is
`[v(i), p(i), p_goal(i) - p(i)]`. With the supplied
`--graph_feat_type "relative"`, its node features are
`[relative_velocity, relative_position, relative_goal, entity_type]`. The
current JAX graph-attention path reverses the first two groups and does not feed
the non-equivariant `entity_type` into graph attention.

Both implementations concatenate the ordinary observation before the selected
agent-node embedding. The original equivalent is in
[`GR_Actor.forward`](InforMARL/onpolicy/algorithms/graph_actor_critic.py#L170):

```python
nbd_features = self.gnn_base(node_obs, adj, agent_id)
actor_features = torch.cat([obs, nbd_features], dim=1)
```

### Action and state transition

The shared current actor call and environment call are in
[`_env_step`](algorithm/marl_ppo.py#L414): the batched observation, graph, and
done mask are passed to `actor_network.apply`; the resulting action is
unbatched and passed to `env.step`. Actions `0..4` represent stay, left, right,
down, and up. [`_discrete_action_to_control_input`](envs/target_mpe_env.py#L215)
converts the action to a unit direction and multiplies it by
`entity_acceleration`.

Ignoring speed clipping, the current integration in
[`_integrate_state`](envs/target_mpe_env.py#L572) is:

```text
p(t+1) = p(t) + v(t) * dt
v(t+1) = (1 - damping) * v(t) + F(t) / mass * dt
```

`F(t)` combines the action force, noise, and collision/environment forces. The
original integration in [`multiagent/core.py`](InforMARL/multiagent/core.py#L278)
updates velocity first:

```text
v(t+1) = (1 - damping) * v(t) + F(t) / mass * dt
p(t+1) = p(t) + v(t+1) * dt
```

Consequently, the current JAX position update uses the old velocity, whereas
the original paper implementation uses the newly updated velocity.

### Reward per agent per step

The current reward is defined in
[`TargetMPEEnvironment.reward`](envs/target_mpe_env.py#L712). With the paper
configuration, every agent receives the same team reward:

```text
r_i(t) = -sum_j ||p_j - goal_j||^2
         - 5 * sum_(j,k) collision(j,k)
         + 5 * number_of_agents_newly_reaching_their_goal
```

The original scenario first gives each agent `-||p_i-goal_i||`, or `+5` when it
is within the goal threshold, and subtracts `5` for each collision. See
[`navigation_graph.py`](InforMARL/multiagent/custom_scenarios/navigation_graph.py#L373).
Because the supplied command leaves `--collaborative` at its default `True`,
[`MultiAgentGraphEnv.step`](InforMARL/multiagent/environment.py#L808) sums those
individual rewards and gives the same total to every agent. Both are therefore
shared team rewards, but the current code uses squared distance and a one-time
goal bonus, while the original uses Euclidean distance and its thresholded goal
reward each step.

### Actor/critic architecture

Both current scripts instantiate `GraphAttentionActorRNN` and `CriticRNN` from
[`model/actor_critic_rnn.py`](model/actor_critic_rnn.py#L112). With the default
network configuration, the actor is:

```text
relative graph features
-> 2 custom Jraph multi-head attention layers (3 heads, hidden size 16)
-> select agent i's node embedding
-> concatenate [o(i), agent-node embedding]
-> Dense(64) + ReLU
-> GRU(64)
-> Dense(64) + ReLU
-> Dense(64) + ReLU
-> Dense(5) categorical logits
```

The original `rmappo` actor uses entity-type embedding and PyTorch Geometric
`TransformerConv`, selects agent `i`'s node output, concatenates
`[o(i), node embedding]`, and then applies an MLP, a 64-unit GRU with LayerNorm,
and a categorical action head. Thus the overall data flow now agrees, but the
attention implementation, feature ordering, entity-type usage, and
normalization are not identical.

The current critic defaults to no graph attention: it embeds entity types,
sums all entity-node features, and applies Dense(64), GRU(64), hidden dense
layers, and a scalar value head. This is not an exact reproduction of the
original graph critic/PopArt setup.

### Configuration differences from the supplied paper command

`python algorithm/marl_ppo.py --paper-config` applies the overrides in
[`with_paper_target_env`](config/mappo_config.py#L192), but it does not reproduce
all settings from the supplied paper command:

| Setting | Current `--paper-config` training | Supplied InforMARL command |
|---|---:|---:|
| Agents | 3 | 3 |
| Parallel environments | 32 | 128 |
| Episode length | 25 | 25 |
| Total environment steps | 2,000,000 | 2,000,000 |
| PPO epochs | 4 | 10 |
| Actor minibatches | 32 | 1 (with automatic target size 128) |
| Learning rate | `5e-4` | `7e-4` |
| Collision penalty coefficient | `-5` in the reward sum | subtract `5` per collision |
| Goal reward | one-time `+5` | `+5` while inside the goal threshold |
| Distance reward | negative squared distance | negative Euclidean distance |
| Actor activation | ReLU | Tanh because `--use_ReLU` is `store_false` |
| Value normalization | None | PopArt enabled |

The agent count, episode length, total environment steps, maximum speed,
visibility radius, relative graph features, and nominal collision penalty are
aligned. The reward equation, integration order, graph network, normalization,
rollout parallelism, and PPO update configuration remain different. Two
original-parser details are easy to miss: `--use_ReLU` is a `store_false`
option and therefore disables ReLU in the policy MLP when explicitly supplied;
`--use_valuenorm` similarly disables ValueNorm, while `--use_popart` enables
PopArt.

### Why the paper-config training run did not converge

The archived W&B run in `wandb_paperTraining.zip` did not merely converge
slowly: its actor became non-finite during the first PPO update. `ratio_0` was
`1` at update 0 and `NaN` for the remaining 487 updates; `ratio`, actor loss,
entropy, and approximate KL were also `NaN`. The critic loss remained finite
and decreased, which isolates the failure to the actor's graph-attention path.

The paper configuration triggers this failure by setting
`add_self_edges_to_nodes=False`. Without landmark self-edges, landmark nodes
have no incoming edges because ordinary graph receivers are agent indices only.
This becomes numerically fatal when combined with the existing invalid graph
padding: the environment reserves `num_agents * num_entities` edge slots and
fills unused sender/receiver indices with `-1`:

```python
valid_agent_idx, valid_entity_idx = jnp.nonzero(
    mask, size=max_num_edge, fill_value=-1
)
```

See [`get_graph`](envs/target_mpe_env.py#L437). These entries are not masked out
before graph batching and attention. In
[`GraphStackedMultiHeadAttention`](model/actor_critic_rnn.py#L281), adding the
per-graph node offset leaves the first graph's padding at `-1` and converts
later graphs' padding into apparently valid indices in preceding graphs. The
result is both invalid input to `segment_softmax` and false cross-environment
edges:

```python
weights = utils.segment_softmax(
    softmax_logits, segment_ids=receivers, num_segments=sum_n_node
)
```

For the first graph, padded receiver `-1` indexes the final node segment, which
is a landmark with no incoming edge under the paper configuration. Its empty
softmax segment has a non-finite maximum/zero normalizer, producing non-finite
graph-attention gradients. After the first bad Adam update, the actor parameters
and optimizer state remain `NaN`; gradient clipping cannot recover them. The
critic stays finite because `use_graph_attention_in_critic` is `False`.

Default training leaves `add_self_edges_to_nodes=True`. The added landmark
self-edges give the final landmark segment a finite softmax normalizer, so the
immediate `NaN` is avoided. This explains why the failure appears with
`--paper-config`. It does not make the default padding correct: later graphs'
padded indices can still become false edges into preceding graphs after offsets
are added. Therefore, disabling self-edges is the configuration trigger, while
unmasked `-1` padding is the underlying implementation defect.

Retraining the paper configuration unchanged is therefore expected to fail
again. Re-enabling self-edges may hide the `NaN`, but changes the intended graph
and does not repair false padded edges. Padded edges must instead be removed or
represented with a valid mask before attention. Temporary finite checks on
actor logits, loss, gradients, and parameters should then verify that the
immediate failure is gone.

After fixing that blocker, remaining convergence differences should be tested
separately: the current run uses 32 environments, 32 minibatches (only three
actor trajectories per minibatch), four PPO epochs, and learning rate `5e-4`,
versus 128 environments, one automatically sized minibatch, ten epochs, and
`7e-4` in the supplied InforMARL command. The squared-distance/one-time-goal
reward, state-integration order, custom graph network, and lack of PopArt also
differ from the original implementation. Finally, `MPELogWrapper` multiplies
returns by the number of agents for reporting; this changes the plotted scale,
but not the reward consumed by PPO.

# List of PPO_Runner_State versions
- v16: First runnable agent, trained on jaxinformarl config, tested on paper config up to 10 agents
- v29:

# My own change to fix v29:
- exp1: prevent padded edges from crossing batched envs
- exp2: reward function to be +5 each step, and negative euclidean distance, and change entity size
+ Replace the inner reward function with:
```python
def _dist_between_target_reward(
        agent_index: Int[Array, AgentIndexAxis], state: MPEState
) -> Float[Array, AgentIndexAxis]:
    corresponding_landmark_index = self.num_agents + agent_index
    distance = jnp.linalg.norm(
        state.entity_positions[agent_index]
        - state.entity_positions[corresponding_landmark_index]
    )

    goal_threshold = 0.1
    return jnp.where(
        distance < goal_threshold,
        5.0,
        -distance,
    )
```
+ remove the current one-time bonus:
```python
one_time_reaching_goal_reward = jnp.sum(
    jax.lax.select(
        state.did_agent_die_this_time_step,
        self.one_time_death_reward,
        jnp.zeros_like(self.one_time_death_reward),
    )
)
```
+ and change to:
```python
return {
    agent_label: global_reward
    for agent_label in self.agent_labels
}
```