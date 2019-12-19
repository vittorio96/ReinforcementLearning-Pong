import gym

#Class that defines the Deep Q learning implementation of the pong game
class PongAgent(object):

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.num_episodes = 1 #number of episodes (games) to play during training

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

    def train(self):
        observation = self.env.reset() #start a game
        n = 0
        while(n < self.num_episodes):
            self.env.render()
            action = self.env.action_space.sample()  # your agent here (this takes random actions)
            observation, reward, done, info = self.env.step(action)

            cur_frame = self.frame_preprocessing(observation)

            if done: #episode (one game) is terminated
                observation = self.env.reset()
                n += 1

        self.env.close()



# Main function to control the agent
def main():
    agent = PongAgent()
    agent.train()


if __name__ == "__main__":
    main()