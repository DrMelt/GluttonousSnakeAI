import shutil
import time
import subprocess
import json
from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from collections import OrderedDict
from collections import deque
import torch.nn.functional as F
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

from snakeGame import SnakeEnv, SnakeVectorEnv
from snake_model import SnakeTransformerModel, SnakeConvModel


is_cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda_available else "cpu")

# -----debug-----
is_show_start_loss = False
is_show_model_out_difrence = False  # Inaccurate when using normalization
is_show_model_weigth_difrence = False
# ----------------

experience_replay_size = int(1024 * 128)
max_size_one_train = int(1024 * 128)
is_train_when_max = False
is_delete_data_after_train = True
new_data_acc = int(0)

is_load_model = True

backup_pre_times = 40
log_pre_times = 10
trained_times = int(0)

env_vector_len = 512

batch_size = 2048
num_epochs = 3
learn_rate = 2e-5
max_grad_norm = 1.0
quality_coef = 1
value_coef = 1
ent_coef = 0.0002
gamma_coef = 0.97
lambda_coef = 0.5

# --------------
model_name = "model_DQN"
model_dir = "models_DQN"


# ----- load model
if is_load_model:
    dqn_net = torch.load(f"./{model_dir}/{model_name}.pth", device)
    dqn_net.train()
else:
    dqn_net = SnakeConvModel().to(device)
    dqn_net.train()

optimizer = optim.Adam(
    dqn_net.parameters(), lr=learn_rate, betas=(0.9, 0.995), eps=1e-6
)
# optimizer = optim.SGD(dueling_net.parameters(), lr=learn_rate)


def save_model():
    torch.save(dqn_net, f"./{model_dir}/{model_name}.pth")


class CustomDataset(Dataset):
    def __init__(self):
        self.actions_ind_list = deque(maxlen=experience_replay_size)

        self.snake_states_list = deque(maxlen=experience_replay_size)
        self.food_states_list = deque(maxlen=experience_replay_size)

        self.rewards_list = deque(maxlen=experience_replay_size)
        self.current_td_scores_list = deque(maxlen=experience_replay_size)
        self.final_mask = deque(maxlen=experience_replay_size)

        self.advances_list = deque(maxlen=experience_replay_size)
        self.quality_fixed_list = deque(maxlen=experience_replay_size)
        self.value_fixed_list = deque(maxlen=experience_replay_size)

        self.ele_num = 0

    def extend(
        self,
        actions_ind_list,
        snake_states_list,
        food_states_list,
        rewords_list,
        final_mask,
    ):
        self.actions_ind_list.extend(actions_ind_list)
        self.snake_states_list.extend(snake_states_list)
        self.food_states_list.extend(food_states_list)
        self.rewards_list.extend(rewords_list)
        self.final_mask.extend(final_mask)

        self.ele_num += len(rewords_list)
        self.ele_num = min(self.ele_num, experience_replay_size)

    def update(self):
        print(f"Avg game len: {len(self.final_mask)/self.final_mask.count(True):.1f}")
        avg_reward = np.array(self.rewards_list).mean()
        print(f"Avg reward: {avg_reward:.4f}")

        with torch.no_grad():
            dqn_net.eval()
            quality_array, value_array = dqn_net.batch_out(
                list(self.snake_states_list),
                list(self.food_states_list),
                batch_size=batch_size,
            )
            dqn_net.train()

        self.current_td_scores_list.clear()
        self.current_td_scores_list.extend(quality_array.tolist())

        # value_array = value_array.clip(min=-2, max=2)
        value_array: np.ndarray = value_array.reshape(-1)

        scores_list = np.zeros((len(self.rewards_list),)).tolist()
        for i in range(len(self.rewards_list) - 1, -1, -1):
            if self.final_mask[i]:
                scores_list[i] = self.rewards_list[i]
            else:
                scores_list[i] = scores_list[i + 1] * gamma_coef + self.rewards_list[i]

        advances_list = np.zeros((len(self.rewards_list),)).tolist()
        for i in range(len(self.rewards_list) - 1, -1, -1):
            if self.final_mask[i]:
                advances_list[i] = scores_list[i] - value_array[i]
            else:
                advances_list[i] = (
                    value_array[i + 1] * gamma_coef
                    + self.rewards_list[i]
                    - value_array[i]
                )

        gae_list = np.zeros((len(self.rewards_list),)).tolist()
        for i in range(len(self.rewards_list) - 1, -1, -1):
            if self.final_mask[i]:
                gae_list[i] = advances_list[i]
            else:
                gae_list[i] = (
                    advances_list[i] + gae_list[i + 1] * lambda_coef * gamma_coef
                )

        gae_array = np.array(gae_list)
        value_gae_array = value_array + gae_array
        value_gae_array = value_gae_array.clip(-2, 2)
        value_fixed_list = value_gae_array.tolist()

        self.advances_list.clear()
        self.advances_list.extend(advances_list)

        self.quality_fixed_list.clear()
        self.quality_fixed_list.extend(gae_list)

        self.value_fixed_list.clear()
        self.value_fixed_list.extend(value_fixed_list)

    def __len__(self):
        return self.ele_num

    def clean(self):
        self.__init__()

    def __getitem__(self, idx):
        action_ind = torch.tensor(self.actions_ind_list[idx], dtype=torch.int64)
        snake_state = torch.tensor(self.snake_states_list[idx], dtype=torch.int64)
        food_state = torch.tensor(self.food_states_list[idx], dtype=torch.int64)

        quality_fixed = torch.tensor(self.quality_fixed_list[idx], dtype=torch.float32)
        value_fixed = torch.tensor(self.value_fixed_list[idx], dtype=torch.float32)

        return (
            action_ind,
            snake_state,
            food_state,
            quality_fixed,
            value_fixed,
        )


dataset = CustomDataset()


def train_net(dataloader):
    # ----- train
    for epoch in range(num_epochs):
        losses_acc = {
            "loss": 0,
            "quality_loss": 0,
            "value_loss": 0,
            "entropy_loss": 0,
        }
        quality_max_min = [-100, 100]
        value_max_min = [-100, 100]

        max_batches = (max_size_one_train - 1) // batch_size + 1
        batch_count = 0

        for (
            actions_ind_batch,
            snake_states_batch,
            food_states_batch,
            quality_fixed_batch,
            value_fixed_batch,
            # advance_target_batch,
        ) in dataloader:

            if batch_count >= max_batches:
                break
            batch_count += 1

            if int(actions_ind_batch.shape[0]) != batch_size:
                continue

            actions_ind_batch = actions_ind_batch.view(-1, 1)
            snake_states_batch = snake_states_batch.view(-1, 1, 100)
            food_states_batch = food_states_batch.view(-1, 1, 100)
            quality_fixed_batch = quality_fixed_batch.view(-1, 1)
            value_fixed_batch = value_fixed_batch.view(-1, 1)

            actions_ind_batch = actions_ind_batch.to(device)
            snake_states_batch = snake_states_batch.to(device)
            food_states_batch = food_states_batch.to(device)
            quality_fixed_batch = quality_fixed_batch.to(device)
            value_fixed_batch = value_fixed_batch.to(device)

            quality, value = dqn_net(
                snake_states_batch,
                food_states_batch,
            )

            action_quality = quality.gather(-1, actions_ind_batch).view(-1, 1)

            quality_loss: torch.Tensor = F.mse_loss(action_quality, quality_fixed_batch)
            value_loss: torch.Tensor = F.mse_loss(value, value_fixed_batch)

            fake_probs: torch.Tensor = F.softmax(quality * 32, dim=-1).view(-1, 4)
            entropy_loss = torch.sum(
                fake_probs * torch.log2(fake_probs + 1e-8), dim=-1
            ).mean()

            loss = (
                quality_loss * quality_coef
                + value_loss * value_coef
                + entropy_loss * ent_coef
            )

            quality_max_min[0] = max(quality_max_min[0], quality.max().item())
            quality_max_min[1] = min(quality_max_min[1], quality.min().item())
            value_max_min[0] = max(value_max_min[0], value.max().item())
            value_max_min[1] = min(value_max_min[1], value.min().item())
            losses_acc["quality_loss"] += quality_loss.item() * quality_coef
            losses_acc["value_loss"] += value_loss.item() * value_coef
            losses_acc["entropy_loss"] += entropy_loss.item() * ent_coef
            losses_acc["loss"] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn_net.parameters(), max_grad_norm)
            optimizer.step()

        losses_acc = {key: value / batch_count for key, value in losses_acc.items()}
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Steps [{batch_count}], Loss: {losses_acc}"
        )
        print(
            f"quality_max_min: {quality_max_min[0]:.4f}  min out: {quality_max_min[1]:.4f}"
        )
        print(
            f"value_max_min :  {value_max_min[0]:.4f}  min out: {value_max_min[1]:.4f}"
        )

    writer.add_scalars("losses", losses_acc, trained_times)


def train_once():

    dqn_net.train()

    global trained_times

    if dataset.__len__() >= experience_replay_size or not is_train_when_max:

        dataset.update()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5
        )

        train_net(dataloader)

        if is_delete_data_after_train:
            dataset.clean()

        trained_times += 1


# def sample_action(obs, random_rate=0.1):
#     if random.random() < random_rate:
#         action_ind = random.randint(0, 3)
#     else:
#         snake_state = torch.tensor(obs[0]).reshape(-1, 1, 100)
#         food_state = torch.tensor(obs[1]).reshape(-1, 1, 100)
#         quality, value = dqn_net(snake_state, food_state)
#         action_ind = quality.argmax().item()
#     return action_ind


def sample_actions(obs_vector, random_rate=0.0):
    obs_tensor = torch.tensor(
        np.array(obs_vector, dtype=np.int64), dtype=torch.int64, device=device
    ).view(len(obs_vector), 2, 100)
    snake_tensor = obs_tensor[:, 0, :]
    food_tensor = obs_tensor[:, 1, :]
    quality, value = dqn_net(snake_tensor, food_tensor)
    quality: torch.Tensor = quality

    # actions_ind = quality.argmax(dim=1).cpu().numpy().tolist()

    # Apply softmax to get probabilities
    probabilities = F.softmax(quality * 256, dim=1)
    # Sample actions based on the probabilities
    actions_ind = (
        torch.multinomial(probabilities, num_samples=1).squeeze().cpu().numpy().tolist()
    )

    for i in range(len(actions_ind)):
        if random.random() < random_rate:
            actions_ind[i] = random.randint(0, 3)

    return actions_ind


def run_game():
    dqn_net.eval()

    vector_env = SnakeVectorEnv(vector_len=env_vector_len)
    obs_vector = vector_env.reset()

    while dataset.__len__() < experience_replay_size:
        actions = sample_actions(obs_vector)

        obs_vector, reward_vector, done_vector, info_vector = vector_env.step(actions)

        dataset.extend(*vector_env.done_data())

    print(f"dataset size: {dataset.__len__()}")
    dqn_net.train()


if __name__ == "__main__":
    while True:
        # ----- get data
        run_game()

        # ----- train
        train_once()

        save_model()
