# imitation-learning-exp

An experiment of behavioral cloning and imitation learning on the highway-env.

## To-do List

- [x] Implement a customized PyTorch dataset to load and sample trajectories (by `torch.utils.data.DataLoader`)
- [x] Read the [deep sets](https://arxiv.org/pdf/1703.06114.pdf) paper and implement it
  - I plan to use deep sets instead of the previously implemented [social attention](https://github.com/KoHomerHu/social-attention-exp/tree/main) because deep sets architecture does not need to learn the query and keys. It may be troublesome if the query and keys are entirely different types of entities (e.g. query is multi-modal sensor data where keys are bounding boxes of detected objects).
  - I actually do not know which one is better, it will be good to test whether deep sets or social attention works best in our own use case. 
- [x] Collect data of manual control and dump it into a pickle file
  - The distribution of actions in `transition_data.pkl` is almost uniform, this should not happen. The API is not recording the manual control action read from the event handler. It could not even be used as a task-irrelevant prior for offline reinforcement learning since the transition is based on manual control action. In short, `transition_data.pkl` is most likely garbage, I'll just leave it there.
- [ ] Experiment on behavioral cloning
  - As an expert data is expected to have a very unbalanced distribution (e.g. most of the time the ego vehicle may be `IDLE`), and BC on discrete action spaces has no difference to a supervised classification task, we may try using [focal loss](https://arxiv.org/abs/1708.02002v2) to alleviate this problem.
- [ ] Experiment on GAIL
- [ ] Read the [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) paper and see if any improvement can be done for GAIL
- [ ] Incorporate GRU / LSTM into GAIL
