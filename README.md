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
| `Actual actor observations (graph-derived)` | The learned 16-value graph embedding passed to `ActorRNN` for action selection. This is the policy's actual observation, rather than the raw six-value environment observation. |
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
observation is passed into `GraphAttentionActorRNN` but is currently replaced
by this graph-derived embedding before `ActorRNN` is called.
