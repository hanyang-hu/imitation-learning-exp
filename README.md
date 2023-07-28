# imitation-learning-exp

An experiment of behavioral cloning and imitation learning on the highway-env.

## To-do List

- [x] Implement a customized PyTorch dataset to load and sample trajectories
- [x] Read the [deep sets](https://arxiv.org/pdf/1703.06114.pdf) paper and implement it
  - we plan to use deep sets instead of the previously implemented [social attention](https://github.com/KoHomerHu/social-attention-exp/tree/main) because deep sets architecture does not need to learn the query and keys. It may be troublesome if the query and keys are entirely different types of entities (e.g. query is multi-modal sensor data where keys are bounding boxes of detect objects).
  - It will be good to test whether deep sets or social attention works best in our own use case. 
- [x] Collect data of manual control and dump it into a pickle file (currently, `transition_data.pkl` contains 11339 state-action pairs)
- [ ] Experiment on behavioral cloning
- [ ] Experiment on GAIL
- [ ] Read the [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) paper and see if any improvement can be done for GAIL
- [ ] Incorporate GRU / LSTM into GAIL
