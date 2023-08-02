# imitation-learning-exp

An experiment of behavioral cloning and imitation learning on the highway-env.

## To-do List

- [x] Implement a customized PyTorch dataset to load and sample trajectories (by `torch.utils.data.DataLoader`)
- [x] Read the [deep sets](https://arxiv.org/pdf/1703.06114.pdf) paper and implement it
  - I plan to experiment deep sets instead of directly using the previously implemented [social attention](https://github.com/KoHomerHu/social-attention-exp/tree/main) because deep sets architecture does not need to learn the query and keys. It may be troublesome if the query and keys are entirely different types of entities (e.g. query is multi-modal sensor data where keys are bounding boxes of detected objects).
  - I actually do not know which one is better, it will be good to test whether deep sets or social attention works best in our own use case. So far I empirically see that deep sets reach less loss (hence higher likelihood) faster compared to social attention, which is what I expected (though I haven't tuned the hyperparameters so that the results could be different).
  - The result is that social attention seems to work better than deep sets. Specifically, it learns to accelerate so that it won't get too hurried when it turns left/right. Impressively 20 episodes with social attention + behavioral cloning achieve performance higher than double DQN in my previous experiment.
- [x] Collect data of manual control and dump it into a pickle file
  - The distribution of actions in `transition_data.pkl` is almost uniform, this should not happen. The API is not recording the manual control action read from the event handler. `transition_data.pkl` is most likely garbage, I deleted it.
  - `transition_data_mc.pkl` would be the one used for training BC.
  - Need to collect manual control data again and figure out how to store and load transition data by episodes, to allow training with GRU / LSTM.
- [x] Experiment on behavioral cloning
  - As an expert data is expected to have a very unbalanced distribution (e.g. most of the time the ego vehicle may be `IDLE`), and BC on discrete action spaces has no difference to a supervised classification task, we may try using [focal loss](https://arxiv.org/abs/1708.02002v2) to alleviate this problem.
  - After 500 epochs over `transition_data_mc.pkl` with a batch size of 128, I tested the agent with 50 episodes, and the average return was $19.13$ (I probably encountered over-fitting with the 1000 epochs setting). 
  - One observed issue of the agent is that it often dies due to crashing when it turns right, I suppose it's because my demonstration data always like to turn right early. As I often speed up, turning right early has no problem for me, but the agent probably didn't learn that. This may be something to notice when collecting more demonstration data.
  - When I re-collected a more "safety-aware" demonstration (`transition_data_mc2.pkl`), the agent's performance goes up to 25.69 on average.
- [ ] Experiment on IQL (with true reward signals)
  - I am considering doing IQL because I may want to test IRL + IQL and I am not that familiar with IRL methods yet (the only one that I know is GAIL), plus the policy extraction step using AWR may be considered somewhere in between behavioral cloning and Q learning (controlled by the hyperparameter $\beta$).
  - Currently, the implementation of IQL does not seem to be correct, the expectile loss of value function quickly descends to 0 whilst the TD loss never decreases.
- [ ] Experiment on offline imitation learning
- [ ] Incorporate GRU / LSTM

## Thoughts about offline imitation learning

My intention for this experiment is to make the agent not interact with the environment during training (i.e. work in an **offline** setting), given a dataset generated by the expert that only contains tuples in the form of $(s, a, s')$ (i.e. no reward signals). I only found some random papers about "offline imitation learning", and haven't really taken time to read them yet.

I want to figure out how to achieve this (BC is clearly an approach, so I am actually looking for improvements). To me, a direct approach is to use IRL to learn a reward function $r_\theta(s, a)$, then relabel the dataset of $(s, a, s')$ pairs into $(s, a, r_\theta(s, a), s')$ and conduct offline reinforcement learning. 

## Results

Returns are averaged in 50 episodes under the default setting of highway-env.

|      Algorithm     |  Mean Return | + GRU |
|:-------------:|:------:|:----:|
|  BC | 25.69 | Pending |
|    IRL   |   Pending | Pending |
| IQL |    Pending | Pending |
| IRL + IQL| Pending | Pending |

## How to use `manual_control.py`

As you can see line 5 of `manual_control.py` imports the `action_listener` method which does not exist in the original code of highway_env:

```from highway_env.envs.common.graphics import action_listener```

I implemented this logic to record the manual control actions as this function is not supported officially.

Specifically, I have added the following parts:

```
current_action = 1
updated = False

def update_action(new_action):
    global current_action, updated
    current_action = new_action
    updated = True

def action_listener():
    global current_action, updated
    if not updated:
        return 1 # if not updated during this iteration, then means no action, i.e. IDLE
    updated = False
    return current_action
```
and modified the `handle_discrete_action_event` method in the `EventHandler` class to:

```
def handle_discrete_action_event(cls, action_type: DiscreteMetaAction, event: pygame.event.EventType) -> None:
    global current_action
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_RIGHT and action_type.longitudinal:
            action_type.act(action_type.actions_indexes["FASTER"])
            update_action(3)
        if event.key == pygame.K_LEFT and action_type.longitudinal:
            action_type.act(action_type.actions_indexes["SLOWER"])
            update_action(4)
        if event.key == pygame.K_DOWN and action_type.lateral:
            action_type.act(action_type.actions_indexes["LANE_RIGHT"])
            update_action(2)
        if event.key == pygame.K_UP:
            action_type.act(action_type.actions_indexes["LANE_LEFT"])
            update_action(0)
```
In order to make use of such modification, run the following script to find out where the folder of `highway_env` is:

```
import highway_env
print(highway_env.__file__)
```
Then direct to the `\highway_env\envs\common\` folder and replace the `graphics.py` with the one in my repo.
