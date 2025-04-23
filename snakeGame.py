import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


IS_EAT_FOOD = 0


class SnakeEnv(gym.Env):

    def __init__(self, grid_size=10, max_food_step=128):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.max_food_step = max_food_step

        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(grid_size, grid_size), dtype=np.int32
        )

        self.food_step = 0

        self.reset()

    def reset(self):
        self.snake = [
            (
                self.grid_size // 2 + random.randint(-2, 2),
                self.grid_size // 2 + random.randint(-2, 2),
            )
        ]
        self.direction = (0, 1)
        self._place_food()
        self.done = False
        self.food_step = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(
            (2, self.grid_size, self.grid_size), dtype=np.int32
        )  # snake, food
        for i, (x, y) in enumerate(self.snake):
            if x >= 0 and x < self.grid_size and y >= 0 and y < self.grid_size:
                obs[0, x, y] = i + 1

        fx, fy = self.food
        obs[1, fx, fy] = 1
        return obs

    def _place_food(self):
        while True:
            self.food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if self.food not in self.snake:
                break

    def step(self, action):
        if action == 0:  # up
            self.direction = (-1, 0)
        elif action == 1:  # down
            self.direction = (1, 0)
        elif action == 2:  # left
            self.direction = (0, -1)
        elif action == 3:  # right
            self.direction = (0, 1)

        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1],
        )
        end_head = self.snake.pop()

        is_eat_food = new_head == self.food
        self.food_step += 1
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
            or new_head in self.snake
            or self.food_step > self.max_food_step
        ):
            self.snake.insert(0, new_head)
            self.done = True
            reward = -2
        else:
            self.snake.insert(0, new_head)
            if is_eat_food:
                self.food_step = 0
                self.snake.append(end_head)
                if len(self.snake) >= self.grid_size * self.grid_size:
                    self.done = True
                    reward = 2
                else:
                    self._place_food()
                    reward = 1
            else:
                reward = -0.02

        obs = self._get_obs()
        info = {IS_EAT_FOOD: is_eat_food}
        return obs, reward, self.done, info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt

        # 获取观测数据
        obs = self._get_obs()
        # 创建一个RGB图像数组
        rgb_obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        rgb_obs[:, :, 0] = np.clip(obs[0], 0, 1) * (
            obs[0] / len(self.snake) * 0.5 + 0.5
        )  # 归一化到0-1

        rgb_obs[:, :, 1] = obs[1]

        if not hasattr(self, "fig"):
            self.fig, self.ax = plt.subplots()

            self.im = self.ax.imshow(rgb_obs)
            plt.ion()  # Turn on interactive mode
            plt.show()
        else:
            self.im.set_data(rgb_obs)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


class SnakeVectorEnv:
    def __init__(self, vector_len=64, max_food_step=128):
        self.vector_len = vector_len

        self.vector_env = [
            SnakeEnv(max_food_step=max_food_step) for _ in range(self.vector_len)
        ]

        self.pre_obs_vector = [None for _ in range(self.vector_len)]

        self.actions_ind_buffer = [[] for _ in range(self.vector_len)]
        self.snake_states_buffer = [[] for _ in range(self.vector_len)]
        self.food_states_buffer = [[] for _ in range(self.vector_len)]
        self.rewords_buffer = [[] for _ in range(self.vector_len)]
        self.final_mask_buffer = [[] for _ in range(self.vector_len)]

        self.actions_ind_list = []
        self.snake_states_list = []
        self.food_states_list = []
        self.rewords_list = []
        self.final_mask_list = []

    def done_data(self):
        (
            actions_ind_list,
            snake_states_list,
            food_states_list,
            rewords_list,
            final_mask_list,
        ) = (
            self.actions_ind_list.copy(),
            self.snake_states_list.copy(),
            self.food_states_list.copy(),
            self.rewords_list.copy(),
            self.final_mask_list.copy(),
        )

        for item in (
            self.actions_ind_list,
            self.snake_states_list,
            self.food_states_list,
            self.rewords_list,
            self.final_mask_list,
        ):
            item.clear()

        return (
            actions_ind_list,
            snake_states_list,
            food_states_list,
            rewords_list,
            final_mask_list,
        )

    def reset(self):
        obs_vector = []

        for i, env in enumerate(self.vector_env):
            obs = env.reset()
            obs_vector.append(obs)
            self.pre_obs_vector[i] = obs
        return obs_vector

    def step(self, actions):
        obs_vector = []
        reward_vector = []
        done_vector = []
        info_vector = []

        for i, env in enumerate(self.vector_env):
            obs, reward, done, info = env.step(actions[i])

            is_end = done

            self.actions_ind_buffer[i].append(actions[i])
            self.snake_states_buffer[i].append(self.pre_obs_vector[i][0])
            self.food_states_buffer[i].append(self.pre_obs_vector[i][1])
            self.rewords_buffer[i].append(reward)
            self.final_mask_buffer[i].append(is_end)

            if is_end:
                self.actions_ind_list.extend(self.actions_ind_buffer[i])
                self.actions_ind_buffer[i].clear()
                self.snake_states_list.extend(self.snake_states_buffer[i])
                self.snake_states_buffer[i].clear()
                self.food_states_list.extend(self.food_states_buffer[i])
                self.food_states_buffer[i].clear()
                self.rewords_list.extend(self.rewords_buffer[i])
                self.rewords_buffer[i].clear()
                self.final_mask_list.extend(self.final_mask_buffer[i])
                self.final_mask_buffer[i].clear()

                obs = env.reset()

            self.pre_obs_vector[i] = obs
            obs_vector.append(obs)
            reward_vector.append(reward)
            done_vector.append(False)
            info_vector.append(info)

        return obs_vector, reward_vector, done_vector, info_vector


if __name__ == "__main__":
    # 测试环境
    env = SnakeEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, info = env.step(action)
        env.render()
