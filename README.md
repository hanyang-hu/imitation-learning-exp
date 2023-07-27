# imitation-learning-exp

An experiment of behavioral cloning and imitation learning on the highway-env.

## To-do List

- [ ] Implement a customized PyTorch dataset to load and sample trajectories
- [ ] Read the [deep sets](https://arxiv.org/pdf/1703.06114.pdf) paper and implement it
  - we plan to use deep sets instead of the previously implemented [social attention](https://github.com/KoHomerHu/social-attention-exp/tree/main) because deep sets can learn a representation separately (which may also be viewed as a con), making the states embedding more stable; plus if the query and keys are entirely different types of entities (e.g. query are multi-modal sensor data where keys are bounding boxes of detect objects) then we have to pray for the attention mechanism to learn embeddings of them that works.
- [ ] Collect data of manual control and dump it into a pickle file
- [ ] Experiment on behavioral cloning
- [ ] Experiment on GAIL
- [ ] Read the [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf) paper and see if any improvement can be done for GAIL
- [ ] Incorporate GRU / LSTM into GAIL
