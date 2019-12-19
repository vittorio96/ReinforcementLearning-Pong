import gym
import keras
import numpy as np

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.max_episodes = 1 #number of episodes (games) to play during training
        self.experience_buffer = None

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

    def build_deep_rl_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(filters = 2, kernel_size=(2,2), strides=1, padding='same', activation='relu', input_shape=(80, 80, 1)))
        model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 40, activation='relu'))
        model.add(keras.layers.Dense(units = 20, activation='relu'))
        model.add(keras.layers.Dense(units = 1, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

        return model

    def train(self):
        cur_frame = self.frame_preprocessing(self.env.reset())
        #prev_frame = np.zeros(shape=(80,80))
        episodes = 0

        #model = self.build_deep_rl_model()

        while(episodes < self.max_episodes):
            self.env.render()

            action = self.env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = self.env.step(action)

            prev_frame = cur_frame
            cur_frame = self.frame_preprocessing(observation)

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