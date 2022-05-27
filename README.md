## Solving Mountain Car with Tile Coding

Linear function approixmation with Q-leanring

### Tile Coding Example
```python
env = gym.make('MountainCar-v0')

features = np.array([env.observation_space.low,env.observation_space.high]).T
delta = env.observation_space.high-env.observation_space.low

num_tiles = 5
width = 10
offset = [i*(delta/width)/num_tiles for i in range(num_tiles)]

tileCoder = TileCoder(features,num_tiles,width,offset,env)

encoded_state = tileCoder.get_one_hot_tiles(state,action)
```

### Stacks

- Ray
- Numpy
