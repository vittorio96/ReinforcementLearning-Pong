import random
import gym
import keras
import numpy as np

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.max_episodes = 50 #number of episodes (games) to play during training
        self.num_actions = 6
        self.epsilon = 0.5
        self.epsilon_decay = 0.98
        self.experience_buffer_size = 800
        self.mini_batch_size = int(self.experience_buffer_size * 0.5)
        self.update_rate = 0.8
        self.update_rate_decay = 0.98
        self.experience_buffer = []
        self.gamma = 0.8
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

    def update_experience_buffer(self, train_sample):

        if len(self.experience_buffer) < self.experience_buffer_size:
            self.experience_buffer.append(train_sample)
        else:
            if random.random() < self.update_rate:
                to_del_idx = random.randint(0, self.experience_buffer_size - 1)
                self.experience_buffer[to_del_idx] = train_sample

    def learn_from_experience(self):
        mini_batch = random.sample(self.experience_buffer, min(self.mini_batch_size, len(self.experience_buffer)))
        self.update_rate *= self.update_rate_decay

        X, y = [], []

        # Prepare the training data
        for cur_state, prev_state, action, reward in mini_batch:
            target = reward + self.gamma * np.amax(self.model.predict(cur_state)[0])
            target_f = self.model.predict(prev_state)
            target_f[0][action] = target
            # Add example to train
            X.append(prev_state.reshape(80, 80, 1))
            y.append(target_f[0])
        X = np.array(X)
        y = np.array(y)

        self.model.fit(x=X, y=y, epochs=10)

    def choose_next_action(self, cur_state):
        act_values = self.model.predict(cur_state)[0]
        action = np.argmax(act_values)
        q_value = np.max(act_values)

        return action, q_value

    def get_random_action(self):
        self.epsilon *= self.epsilon_decay
        return random.randrange(self.num_actions)

    def build_deep_rl_model(self):

        model = keras.models.Sequential()
        model.add(keras.layers.convolutional.Conv2D(filters = 2, kernel_size=(2, 2), strides=1, padding='same', activation='relu', input_shape=(80, 80, 1)))
        model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units = 40, activation='relu'))
        model.add(keras.layers.Dense(units = 20, activation='relu'))
        model.add(keras.layers.Dense(units = self.num_actions, activation='sigmoid'))

        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self):
        episodes = 0
        gamma = 1

        prev_frame = np.zeros(shape=(80,80))
        cur_frame = self.frame_preprocessing(self.env.reset())
        prev_state = None
        cur_state = (cur_frame - prev_frame).reshape((-1, 80, 80, 1))


        while(episodes < self.max_episodes):
            self.env.render()

            if np.random.rand() <= self.epsilon:
                action =  random.randrange(self.num_actions)
            else:
                action, q_value = self.choose_next_action(cur_state)

            observation, reward, done, info = self.env.step(action)

            prev_frame = cur_frame
            cur_frame = self.frame_preprocessing(observation)
            prev_state = cur_state
            cur_state = (cur_frame - prev_frame).reshape((-1, 80, 80, 1))

            if done: #episode (one game) is terminated
                episodes += 1
                cur_frame = self.frame_preprocessing(self.env.reset())
                self.learn_from_experience()
            else:
                train_sample = (cur_state, prev_state, action, reward)
                self.update_experience_buffer(train_sample)

        self.env.close()


# Main function to control the agent
def main():
    agent = PongAgent()
    agent.train()

# Invoke main
if __name__ == "__main__":
    main()