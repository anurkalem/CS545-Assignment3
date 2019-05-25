import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

from Environment import PongEnvironment

TEST_COUNT = 10
TEST_EPISODE_COUNT = 100

LR = 1e-4

ENTROPY_LOSS = 1e-3
LOSS_CLIPPING = 0.2


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

    return loss


class Agent:

    def __init__(self, env):
        self.environment = env

        self.action_size = 3
        self.state_size = 2

        self.actor = self.build_actor()
        self.actor.load_weights("PongModel.h5")

    def build_actor(self):
        state_input = Input(shape=(self.state_size,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,))

        x = Dense(128, activation='tanh')(state_input)
        x = Dense(128, activation='tanh')(x)
        out_actions = Dense(self.action_size, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR), loss=[proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction)])

        return model

    def get_action(self, observation):
        # Generate predict values
        obs = observation.reshape(1, self.state_size)
        advantage_values = np.zeros((1, 1))
        predictions = np.zeros((1, self.action_size))

        # Get predict values from actor
        predicts = self.actor.predict([obs, advantage_values, predictions])

        return np.argmax(predicts[0])

    def play(self, episode):
        obs = self.environment.reset()

        while True:
            action = self.get_action(obs)
            next_obs, _, is_done, = self.environment.step(action)

            if episode == TEST_EPISODE_COUNT - 1:
                self.environment.render()

            obs = next_obs

            if is_done:
                break

    def test(self):
        for episode in range(TEST_EPISODE_COUNT):
            self.play(episode)


if __name__ == '__main__':
    agent = Agent(PongEnvironment(True))

    for test in range(TEST_COUNT):
        agent.test()
