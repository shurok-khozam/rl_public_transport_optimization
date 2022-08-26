from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from random import randint, sample

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  #   0.995
        self.learning_rate = 0.01  #   0.001
        self.tau = .125
        self.batch_size = 8
        self.memory = deque(maxlen=self.batch_size)

        # model is updated instantly
        # target_model updated after each batch
        self.model = self.create_model()
        self.target_model = self.create_model()

    # model network weights predictions
    def create_model(self):
        model = Sequential()
        state_shape = self.env.INPUT_SHAPE

        model.add(Dense(state_shape[0] * state_shape[1], input_dim= 1, activation="relu"))
        model.add(Dense(48 * 4, activation="relu"))
        model.add(Dense(56 * 4, activation="relu"))
        model.add(Dense(100 * 4, activation="relu"))
        model.add(Dense(56 * 4, activation="relu"))
        model.add(Dense(self.env.OUTPUT_SHAPE))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def flatten_state(self, state):
        return state.flatten()

    def action(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        print("<------> New Epsilon value: " + str(self.epsilon))
        action = 0
        if np.random.random() < self.epsilon:
            # Exploration
            action = int(self.env.ACTIONS[randint(0, len(self.env.ACTIONS)-1)])
            print("------> Taking action: Random")
        else:
            # Exploitation
            predictions = self.target_model.predict(self.flatten_state(state))
            # predictions[0] ==> because the output shape is (1, 81) meaning one line and 81 column
            #   so to get the probabilities of taking each of the 81 actions we use the first line (index 0)
            action = np.argmax(predictions[0])
            print("------> Taking action: Predicted")
        print("<------ Taking action: " + str(action) + " => " + str(self.env.MAPPING[self.env.ACTIONS[action]]))
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        # transfer old model of target model after modification after 8 step
        if len(self.memory) < self.batch_size:
            return False
        # each element of memory is cur_state, action, reward, new_state, done(after each step)
        samples = sample(self.memory, self.batch_size)
        for _sample in samples:
            state, action, reward, new_state, done = _sample
            target = self.target_model.predict(self.flatten_state(state))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(self.flatten_state(new_state))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(self.flatten_state(state), target, epochs=1, verbose=0)
        self.clear_memory()
        return True

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def clear_memory(self):
        self.memory.clear()

    def save_model(self, fn):
        self.target_model.save(fn)

    def load_model(self, fn):
        self.target_model = load_model(fn)
