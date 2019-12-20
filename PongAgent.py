import random
import gym
import keras
import numpy as np

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.max_episodes = 1 #number of episodes (games) to play during training
        self.num_actions = 6
        self.epsilon = 0.5
        self.epsilon_decay = 0.98
        self.episode_buffer = None
        self.experience_buffer = None
        self.model = self.build_deep_rl_model()

    def frame_preprocessing(self, frame_img):
        # Remove redundant pixels from the image (e.g. points and white line) and downsample with a factor of two
        frame_img = frame_img[35:195]
        frame_img = frame_img[0::2, 0::2, 0] #takes one element every two elements
        # Create a uniform background color
        frame_img[frame_img == 144] = 0
        frame_img[frame_img == 109] = 0
        # Set balls and paddles to the same color
        frame_img[frame_img != 0] = 1

        return frame_img #80x80 numpy array

    def update_experience_buffer(self):

        return None

    def choose_next_action(self, cur_state):
        act_values = self.model.predict(cur_state)[0]
        action = np.argmax(act_values)
        q_value = np.max(act_values)

        return action, q_value

    def store_in_episode_buffer(self):
        self

    def build_deep_rl_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(filters = 2, kernel_size=(2, 2), strides=1, padding='same', activation='relu', input_shape=(80, 80, 1)))
        model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 40, activation='relu'))
        model.add(keras.layers.Dense(units = 20, activation='relu'))
        model.add(keras.layers.Dense(units = self.num_actions, activation='softmax'))

        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self):
        episodes = 0
        gamma = 1

        prev_frame = np.zeros(shape=(80,80))
        cur_frame = self.frame_preprocessing(self.env.reset())
        cur_state = (cur_frame - prev_frame).reshape((-1, 80, 80, 1))

        while(episodes < self.max_episodes):
            self.env.render()

            #action = self.env.action_space.sample()  # your agent here (this takes random actions)
            action, q_value = self.choose_next_action(cur_state)
            observation, reward, done, info = self.env.step(action)

            prev_frame = cur_frame
            cur_frame = self.frame_preprocessing(observation)

            cur_state = (cur_frame-prev_frame).reshape((-1, 80, 80, 1))
            next_action, next_q_value = self.choose_next_action(cur_state)
            target = reward + gamma * next_q_value

            if done: #episode (one game) is terminated
                cur_frame = self.frame_preprocessing(self.env.reset())
                episodes += 1

        self.env.close()


# Main function to control the agent
def main():
    agent = PongAgent()
    agent.train()

# Invoke main
if __name__ == "__main__":
    main()