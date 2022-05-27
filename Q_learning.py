import numpy as np
class Q_learning:

    def __init__(self, state_size, action_size, tileCoder, config):

        self.w = np.random.uniform(-1, 1, size=(state_size * action_size, 1))
        self.num_actions = action_size

        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        self.tileCoder = tileCoder


    def update(self, encoded_sa, next_state, action, reward, done):
        '''
        state : List[]
        '''
        current_action_value = np.einsum('ij,ij->j', encoded_sa, self.w)

        next_action_value = np.max(
            [np.einsum('ij,ij->j', self.tileCoder.get_one_hot_tiles(next_state, a), self.w) for a in range(self.num_actions)])

        td_error = (reward + done * self.gamma * next_action_value - current_action_value)

        self.w += self.learning_rate * (td_error * encoded_sa)

        return td_error

    def on_epoch_end(self):
        pass