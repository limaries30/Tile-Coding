import numpy as np

class TileCoder:

    def __init__(self, features, num_tiles, width, offset, env):
        self.total_states = (width ** 2) * num_tiles
        self.num_tiles = num_tiles
        self.offset = offset
        self.width = width
        self.num_features = len(features)
        self.num_actions = env.action_space.n
        self.tiles = self.generate_tiles(features, width)

    def discretize(self, feature):
        return np.linspace(feature[0], feature[1], self.width + 1)[1:-1]

    def generate_tiles(self, features, width):
        '''
        only called at initialization
        return [tile_1:[feature_1_tile,feature_2_tile]...]
        '''
        tiles = []

        discretiezed_features = np.array(list(map(lambda x: self.discretize(x), features)))  # 2*(width-1)
        tiles = np.array(
            list(map(lambda x: discretiezed_features + x.reshape(2, 1), self.offset)))  # NUM_TIELS*2*(width-1)

        return tiles

    def decode(self, state)->List[int]:
        '''
        input : feature from gym
        return List:[[tile_1_feature_1,tile_1_feature_2],...]
        '''

        decoded_features = np.array([np.digitize(s, t) for tile in self.tiles for s, t in zip(state, tile)]).reshape(
            self.num_tiles, 2)

        return decoded_features

    def get_one_hot_tiles(self, state, action)->List[int]:
        '''
        return one_hot_encoded_list
        '''
        decoded_features = self.decode(state)

        result = [tile_coord[0] + tile_coord[1] * self.width + (self.width ** 2) * idx + self.total_states * action for
                  idx, tile_coord in enumerate(decoded_features)]
        one_hot_vector = np.zeros((self.total_states * self.num_actions, 1))
        one_hot_vector[result] = 1
        return one_hot_vector