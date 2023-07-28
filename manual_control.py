import gymnasium as gym
import random
import os
import pickle
from highway_env.envs.common.graphics import action_listener

env = gym.make('highway-v0', render_mode = 'rgb_array')
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 40,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False
    },
    "manual_control": True
}
env.configure(config)
env.config["duration"] = 60

filename = './data/transition_data_mc.pkl'
data = []

if os.path.exists(filename):
    data = pickle.load(open(filename, 'rb'))

current_action = 1
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

i = 0
while i < 100:
    density = 1.0 + random.random() * 0.5
    env.config["vehicles_density"] = density # uniformly sampled from [1.0, 1.5]
    i += 1
    print("Round {} of data collection, density is {}.".format(i, density))

    state, _ = env.reset()
    done = truncated = False
    state_action_pair = []

    cnt = 0
    while not (done or truncated):
        # print(state[0])
        next_state, reward, done, truncated, info = env.step(env.action_space.sample())
        current_action = action_listener()
        state_action_pair.append({'state': state, 'action': current_action, 
                                  'reward': reward, 'next_state': next_state})
        # print(ACTIONS_ALL[current_action])
        state = next_state
        env.render()
        cnt += 1

    print(cnt)
    if cnt >= 45:
        print("This episode is added to the dataset!!!")
        data = data + state_action_pair
    else:
        print("This episode is not added to the dataset!!!")
        i -= 1

pickle.dump(data, open(filename, 'wb'))

print("Data is stored!!!")
