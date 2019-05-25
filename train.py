import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

from Environment import PongEnvironment

EPISODE_COUNT = 1000

GAMMA = 0.99
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
        self.critic = self.build_critic()

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

    def build_critic(self):
        state_input = Input(shape=(self.state_size,))

        x = Dense(128, activation='tanh')(state_input)
        x = Dense(128, activation='tanh')(x)
        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def get_action(self, observation):
        # Generate predict values
        obs = observation.reshape(1, self.state_size)
        advantage_values = np.zeros((1, 1))
        predictions = np.zeros((1, self.action_size))

        # Get predict values from actor
        predicts = self.actor.predict([obs, advantage_values, predictions])

        # Get max action from predict values
        action = np.random.choice(self.action_size, p=np.nan_to_num(predicts[0]))

        # Generate action matrix for action
        action_matrix = np.zeros(self.action_size)
        action_matrix[action] = 1

        return action, action_matrix, predicts

    def get_batch(self, episode):
        S = []
        A = []
        R = []
        P = []

        obs = self.environment.reset()

        while True:
            action, action_matrix, predicts = self.get_action(obs)
            next_obs, reward, is_done, = self.environment.step(action)

            S.append(obs)
            A.append(action_matrix)
            R.append(reward)
            P.append(predicts)

            obs = next_obs

            if episode % 50 == 0:
                self.environment.render()

            if is_done:
                self.transform_rewards(R)
                total_rewards = np.sum(R)

                print("Total reward is {} in episode {}".format(total_rewards, episode))

                break

        obs, action, reward, pred = np.array(S), np.array(A), np.reshape(np.array(R), (len(R), 1)), np.array(P)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))

        return obs, action, reward, pred

    def transform_rewards(self, rewards):
        for i in range(len(rewards) - 1, 0, -1):
            rewards[i - 1] += rewards[i] * GAMMA

    def run(self):
        for episode in range(EPISODE_COUNT):
            # Get batch from environment
            obs, actions, rewards, predictions = self.get_batch(episode)
            batch_size = len(obs)

            # Calculate advantage function
            predicted_values = self.critic.predict(obs)
            advantage_values = rewards - predicted_values

            # Train actor and critic according to batch
            self.actor.fit([obs, advantage_values, predictions], [actions], batch_size=batch_size, shuffle=True, epochs=10, verbose=False)
            self.critic.fit([obs], [rewards], batch_size=batch_size, shuffle=True, epochs=10, verbose=False)

            # Save actor parameters when convergence
            if episode % 5 == 0:
                self.actor.save_weights("PongModel.h5", True)


if __name__ == '__main__':
    agent = Agent(PongEnvironment(True))
    agent.run()
