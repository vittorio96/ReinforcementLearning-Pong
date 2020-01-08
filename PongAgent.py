import random
import gym
import keras
import numpy as np
#from scipy.misc import toimage

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.max_episodes = 100 #number of episodes (games) to play during training
        self.num_actions = 6
        self.frames_to_merge = 3
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.experience_buffer_size = 2000
        self.mini_batch_size = int(self.experience_buffer_size * 0.005)
        self.buffer_update_rate = 0.5
        self.buffer_update_rate_decay = 0.96
        self.experience_buffer = []
        self.gamma = 0.65
        self.prediction_model = self.build_deep_rl_model()
        self.target_model = self.build_deep_rl_model()

    def frame_preprocessing(self, frame_img):
        # Remove redundant pixels from the image (e.g. points and white line) and downsample with a factor of two
        frame_img = frame_img[35:195]
        frame_img = frame_img[0::2, 0::2, 0] #takes one element every two elements
        return frame_img #80x80 numpy array

    def update_experience_buffer(self, train_sample):
        # Update the experience buffer
        if len(self.experience_buffer) < self.experience_buffer_size:
            self.experience_buffer.append(train_sample)
        else:
            # Toss a coin to decide whether to add the last example to the experience replay buffer or not
            if (random.random() < self.buffer_update_rate or train_sample[3] > 0):
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
            target_f[0][action] = target
            # Add example to train
            X.append(state.reshape(80, 80, 1))
            y.append(target_f[0])
        X = np.array(X)
        y = np.array(y)
        # Train the model on the last examples
        self.prediction_model.fit(x=X, y=y, epochs=1, verbose=0)
        # Decrement the update rate. Update rate is never below 0.25
        if self.buffer_update_rate_decay > 0.35:
            self.buffer_update_rate *= self.buffer_update_rate_decay

    def choose_next_action(self, cur_state):
        act_values = self.prediction_model.predict(cur_state)[0]
        action = np.argmax(act_values)
        q_value = np.max(act_values)
        return action, q_value

    def get_random_action(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay
        return random.randrange(self.num_actions)

    def build_deep_rl_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(filters = 8, kernel_size=(8, 8), strides=4, padding='same', activation='relu', input_shape=(80, 80, 1)))
        model.add(keras.layers.convolutional.Conv2D(filters=16, kernel_size=(4, 4), strides=2, padding='same', activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 128, activation='relu'))
        model.add(keras.layers.Dense(units = self.num_actions, activation='relu'))

        model.compile(optimizer='adam', loss='mse')
        #model.load_weights("prediction_model_weights_1k.h5")
        return model

    def train(self):
        episodes = 0
        episode_reward_history = []
        average_reward_history = []

        state = self.frame_preprocessing(self.env.reset())

        while(episodes < self.max_episodes):

            #self.env.render()#render a frame

            # Sample an action (exploration vs exploitation)
            if np.random.rand() <= self.epsilon:
                action = self.get_random_action()
            else:
                action, q_value = self.choose_next_action(state.reshape(-1, 80, 80, 1))

            observation, reward, done, info = self.env.step(action)

            # Merge consecutive frames to capture movement
            reward_sum = reward
            next_state = self.frame_preprocessing(observation)

            for _ in range(self.frames_to_merge - 1):
                observation, reward, done, info = self.env.step(action)
                reward_sum += reward
                next_state += self.frame_preprocessing(observation)
                if done:
                    break
                #self.env.render() # render a frame

            avg_frame_reward = reward_sum/self.frames_to_merge
            # Reward normalization to give a stable reward over time
            if avg_frame_reward < 0:
                final_reward = -1
            elif avg_frame_reward > 0:
                final_reward = +1
            else:
                final_reward = 0

            episode_reward_history.append(final_reward)

            #toimage(next_state).show()
            # Prepare for the next transition in the game
            if done: #episode (one game) is terminated
                episodes += 1
                #Analyze average reward
                average_reward_history.append(sum(episode_reward_history) / len(episode_reward_history))
                moving_average = sum(average_reward_history[-min(100, len(average_reward_history)):])/min(100, len(average_reward_history))
                print("Sliding average reward after episode:"+str(moving_average))
                state = self.frame_preprocessing(self.env.reset())
                #Copy parameters from prediction network to target network
                if episodes % 2 == 0:
                    self.target_model.set_weights(self.prediction_model.get_weights())
            else:
                train_sample = (next_state.reshape(-1, 80, 80, 1), state.reshape(-1, 80, 80, 1), action, final_reward)
                self.update_experience_buffer(train_sample)
                self.learn_from_experience()
                state = next_state

        print("Sliding average reward history during training:")
        print(average_reward_history)
        self.prediction_model.save_weights("prediction_model_weights_100.h5")
        self.env.close()

# Main function to control the agent
agent = PongAgent()
agent.train()