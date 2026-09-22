[**Main scripts**](#main_scripts) | [**Typical train and test flow**](#train_test_flow) |  [**Citation**](#citation)

<h2> Results </h2>
<h4>Original JAX implementation - Trained with 3 agent and executed with 10 agents</h4>
<img src="https://github.com/jselvaraaj/JaxInforMARL/blob/main/10_agents.gif?raw=true" alt="Original 10 agents" width="60%">

<h4>My modified JAX implementation - Trained with 3 agent and executed with 10 agents, with the same large world size of 10</h4>
<img src="https://github.com/viethoangle-key/JaxInforMARL/blob/paper-aligned-training/artifacts_paper-aligned-training/10_agents_withObs_0_large.gif?raw=true" alt="My 10 agents" width="60%">

<h4>My modified JAX implementation - Trained with 3 agent and executed with 10 agents, using a smaller world size of 3</h4>
<img src="https://github.com/viethoangle-key/JaxInforMARL/blob/paper-aligned-training/artifacts_paper-aligned-training/10_agents_withObs_0.gif?raw=true" alt="My 10 agents" width="60%">


The original JAX implementation at [**JaxInfoMARL**](#jaxinfomarl) was developed to greatly speed up the training process of the original [**InfoMARL**](#infomarl) paper. However, in their example, most of the agents could not reach the goal and just hovers around the space instead. In my repository, with the same large world size, almost all agents reach the goal proximity. With a smaller world size of 3 similar to InfoMARL environment configuration, agents reach fully inside goal with 100% success rate. In order to achieve this, I modified `JaxInfoMARL` to include more critical features in state design and reward design that were implemented for a scalable RL system in [**InfoMARL**](#infomarl), including:
- Fixed a bug at which the global position observation of agent was always [0, 0], and concatenate observations with aggregated local information to reproduce the full InfoMARL's state design.
- Used goal reward per step like the original InfoMARL paper, in order to encourage reaching goal at the earliest instead of one-time goal reward, while increasing `collision_reward_coefficient=-5.0` same as InfoMARL's large negative collision coefficient.


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

<h2 name="citation" id="citation">Citation</h2>

<r name="infomarl" id="infomarl">Original InfoMARL paper and implementation:</r>

```
@software{InforMARL,
      title={InforMARL: Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation},
      author={Nayak et al},
      url = {https://github.com/nsidn98/InforMARL},
    }
```

<r name="jaxinfomarl" id="jaxinfomarl">JaxInforMARL implementation:</r>

```
@software{JaxInforMARL,
      title={JaxInforMARL: Multi-Agent Target MPE RL Environments with GNNs in JAX},
      author={Joseph Selvaraaj},
      year = {2025},
      url = {https://github.com/jselvaraaj/JaxInforMARL},
      version = {1.0.0}
    }
```
