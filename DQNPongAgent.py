import random
import gym
import keras
import numpy as np

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.max_episodes = 200 #number of episodes (games) to play during training
        self.num_actions = 2
        self.frames_to_merge = 3
        self.epsilon = 0.95
        self.epsilon_decay = 0.99
        self.experience_buffer_size = 4000
        self.mini_batch_size = 32
        self.experience_buffer = []
        self.gamma = 0.96
        self.prediction_model = self.build_deep_rl_model()
        self.target_model = self.build_deep_rl_model()

    def frame_preprocessing(self, frame_img):
        # Remove redundant pixels from the image (e.g. points and white line) and downsample with a factor of two
        frame_img = frame_img[35:195]
        frame_img = frame_img[::4, ::4, 0]  # takes one element every four elements
        frame_img[frame_img == 144] = 0
        frame_img[frame_img == 109] = 0
        frame_img[frame_img != 0] = 255
        return frame_img/255  # 40x40 numpy array

    def update_experience_buffer(self, train_sample):
        # Update the experience buffer
        if len(self.experience_buffer) < self.experience_buffer_size:
            self.experience_buffer.append(train_sample)
        else:
            to_del_idx = random.randint(0, self.experience_buffer_size - 1)
            self.experience_buffer[to_del_idx] = train_sample

    def learn_from_experience(self):
        # Build a random mini batch of examples from the experience buffer
        mini_batch = random.sample(self.experience_buffer, min(self.mini_batch_size, len(self.experience_buffer)))
        X, y = [], []
        # Prepare the training data
        for next_state, state, action, reward in mini_batch:
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.prediction_model.predict(state)
            target_f[0][action - 2] = target
            # Add example to train
            X.append(state)
            y.append(target_f)
        X = np.array(X[0])
        y = np.array(y[0])
        # Train the model on the last examples
        self.prediction_model.fit(x=X, y=y, epochs=1, verbose=0)

    def choose_next_action(self, cur_state):
        act_values = self.prediction_model.predict(cur_state)[0]
        action = np.argmax(act_values) + 2
        q_value = np.max(act_values)
        return action, q_value

    def get_random_action(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay
        return random.randrange(self.num_actions) + 2

    def build_deep_rl_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(filters = 32, kernel_size=(4, 4), strides=2, padding='same', activation='relu', input_shape=(40, 40, 3)))
        model.add(keras.layers.convolutional.Conv2D(filters=64, kernel_size=(4, 4), strides=2, padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.Dense(units = self.num_actions, activation='linear'))

        model.compile(optimizer='adam', loss='mse')
        #model.load_weights("prediction_model_weights_300.h5")
        return model

    def train(self):
        episodes = 0
        episode_reward_history = []
        average_reward_history = []

        initial_state = self.frame_preprocessing(self.env.reset())
        state = np.stack((initial_state, initial_state, initial_state), axis = 2)
        state = state.reshape(1, 40, 40, 3)
        goals_done = 0

        while episodes < self.max_episodes:

            #self.env.render()#render a frame

            # Sample an action (exploration vs exploitation)
            if np.random.rand() <= self.epsilon:
                action = self.get_random_action()
            else:
                action, q_value = self.choose_next_action(state)

            observation, reward, done, info = self.env.step(action)

            # Prepare the next state by merging consecutive frames
            obs = self.frame_preprocessing(observation)
            obs = obs.reshape(1, 40, 40, 1)
            next_state = np.append(obs, state[:,:,:,:self.frames_to_merge-1], axis = 3)

            # Measure learning metrics
            if reward != 0: episode_reward_history.append(reward)
            if reward == 1: goals_done += 1

            # Prepare for the next transition in the game
            if done: #episode (one game) is terminated
                episodes += 1
                #Analyze average reward
                average_reward_history.append(sum(episode_reward_history) / len(episode_reward_history))
                moving_average = sum(average_reward_history[-min(12, len(average_reward_history)):])/min(12, len(average_reward_history))
                print("Sliding average reward after episode:"+str(moving_average))
                print("Number goals done:" + str(goals_done))
                #Copy parameters from prediction network to target network
                if episodes % 4 == 0:
                    self.target_model.set_weights(self.prediction_model.get_weights())
                    self.prediction_model.save_weights("prediction_model_weights_{}.h5".format(episodes))
                #Initialize new episode
                initial_state = self.frame_preprocessing(self.env.reset())
                state = np.stack((initial_state, initial_state, initial_state), axis = 2)
                state = state.reshape(1, 40, 40, 3)
                goals_done = 0
            else:
                train_sample = (next_state, state, action, reward)
                self.update_experience_buffer(train_sample)
                self.learn_from_experience()
                state = next_state

        print("Sliding average reward history during training:")
        print(average_reward_history)
        self.prediction_model.save_weights("prediction_model_weights_200.h5")
        self.env.close()

# Main function to control the agent
agent = PongAgent()
agent.train()