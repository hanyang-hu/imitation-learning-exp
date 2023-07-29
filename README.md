# imitation-learning-exp

An experiment of behavioral cloning and imitation learning on the highway-env.

## To-do List

- [x] Implement a customized PyTorch dataset to load and sample trajectories (by `torch.utils.data.DataLoader`)
- [x] Read the [deep sets](https://arxiv.org/pdf/1703.06114.pdf) paper and implement it
  - I plan to use deep sets instead of the previously implemented [social attention](https://github.com/KoHomerHu/social-attention-exp/tree/main) because deep sets architecture does not need to learn the query and keys. It may be troublesome if the query and keys are entirely different types of entities (e.g. query is multi-modal sensor data where keys are bounding boxes of detected objects).
  - I actually do not know which one is better, it will be good to test whether deep sets or social attention works best in our own use case. So far I empirically see that deep sets reach less loss (hence higher likelihood) faster compared to social attention, which is what I expected (though I haven't tuned the hyperparameters so that the results could be different). Performance-wise, sadly, both architectures only learned to be IDLE for behavioral cloning.
  - Funny thing for the last bullet point: I actually forgot to turn off manual control :), no wonder why they are all IDLE. The result is that social attention seems to work better than deep sets (again, I didn't tune much hyperparameters so you know), specifically it learns to achieve faster speed so that when it turns it would not turn too hurry (I guess so? Please don't trust too much on this observation as I might take it back after more experiments). Imprssively 20 episodes with social attention + behavioral cloning achieve performance higher than double DQN in my previous experiment.
- [x] Collect data of manual control and dump it into a pickle file
  - The distribution of actions in `transition_data.pkl` is almost uniform, this should not happen. The API is not recording the manual control action read from the event handler. `transition_data.pkl` is most likely garbage, I'll just leave it there.
  - `transition_data_mc.pkl` would be the one used for training BC.
- [x] Experiment on behavioral cloning
  - As an expert data is expected to have a very unbalanced distribution (e.g. most of the time the ego vehicle may be `IDLE`), and BC on discrete action spaces has no difference to a supervised classification task, we may try using [focal loss](https://arxiv.org/abs/1708.02002v2) to alleviate this problem.
- [ ] Experiment on GAIL
- [ ] Read the [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) paper and see if any improvement can be made for GAIL
- [ ] Incorporate GRU / LSTM into GAIL

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
