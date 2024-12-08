import os
import time
import imageio
import torch
import torch.nn as nn
import torch.distributions as distributions
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from envs.mujoco_env import MujocoEnv
from utils.misc import as_numpy, as_tensor, get_device, init_mlp
from dm_env import specs, TimeStep, StepType



class Actor(nn.Module):
    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        log_std = -0.05 * torch.ones(network_sizes[-1], dtype=torch.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = init_mlp(network_sizes)
        self.buffers = {"logp_a_history": [], "episode_rewards": []}
        self.to(self.device)

    def forward(self, observation):
        observation = as_tensor(observation, device=self.device)
        policy = distributions.Normal(
            loc=self.mu_net(observation), scale=torch.exp(self.log_std)
        )
        return policy

    def get_action(self, policy):
        # Note: Values of logp_a are lower than I initially expected
        #       (with exp(logp_a) much less than .9-1.0). Recall that if you
        #       plot Normal(0,1) the peak probability density is about .40
        #       This is why the values are lower than I thought they would be
        action = policy.rsample()
        logp_a = policy.log_prob(action)
        self.buffers["logp_a_history"].append(logp_a)
        return action, logp_a

    def clear_buffers(self):
        self.buffers = {"logp_a_history": [], "episode_rewards": []}

    def __getitem__(self, key):
        return self.buffers[key]


class Critic(nn.Module):
    def __init__(self, network_sizes, device=None):
        super().__init__()
        self.device = get_device(device=device)
        self.value_approximator = init_mlp(network_sizes)
        self.buffers = {"value_history": []}
        self.to(self.device)

    def forward(self, observation):
        observation = as_tensor(observation, device=self.device)
        value = self.value_approximator(observation)
        self.buffers["value_history"].append(value)
        return value

    def clear_buffers(self):
        self.buffers = {"value_history": []}

    def __getitem__(self, key):
        return self.buffers[key]


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        model_dir,
        hidden_sizes=(256, 256),
        lr=3e-4,
        gamma=0.99,
        device=None,
    ):
        super().__init__()
        self.device = get_device(device=device)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.gamma = gamma

        actor_sizes = (observation_space.shape[0], *hidden_sizes, action_space.shape[0])
        super(ActorCritic, self).add_module(
            "actor", Actor(network_sizes=actor_sizes, device=device),
        )
        critic_sizes = (
            observation_space.shape[0],
            *hidden_sizes,
            1,
        )
        super(ActorCritic, self).add_module(
            "critic", Critic(network_sizes=critic_sizes, device=device)
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)
        self.accumulated_rewards = []

    def forward(self, observation):
        observation = as_tensor(observation, device=self.device)
        value = self.critic(observation)
        policy = self.actor(observation)
        return value, policy

    def update(self):
        self.optim.zero_grad()

        reward_to_go = 0
        rewards = []

        # Calculate Extended Advantage Estimate
        for reward in reversed(self.actor["episode_rewards"]):
            reward_to_go = reward + (self.gamma * reward_to_go)
            rewards.insert(0, reward_to_go)

        # Convert values to tensors
        q_t = as_tensor(rewards, device=self.device)
        value_history = torch.Tensor(self.critic["value_history"]).to(self.device)
        logp_a_history = torch.stack(self.actor["logp_a_history"]).to(self.device)

        # Calcualte loss of each module
        advantage = q_t - value_history  # Q(s, a) - V(s)
        critic_loss = advantage.pow(
            2
        ).mean()  # Minimize advantage**2, acting optimally means that advantage should be 0
        actor_loss = (
            -logp_a_history.sum(axis=1) * advantage
        ).mean()  # ∇_θ(𝓙(π_θ)) = 𝔼[∇_θ(log π_θ(a_t | s_t) * A(s_t, a_t))]

        # Use gradient descent
        loss = critic_loss + actor_loss
        loss.backward()
        self.optim.step()

        # Reset buffers for actor/critic
        self.clear_buffers()
        return actor_loss, critic_loss

    def train(self, env:MujocoEnv, steps):
        state: TimeStep = env.reset()
        stats = {"reward_history": [], "pi_loss": [], "v_loss": []}
        max_reward = -1000000000000
        episode_num = 0

        video_step_counter = 0
        video_creator = []


        for step in tqdm(range(steps)):
            value, policy = self(state.observation)
            action, logp_a = self.actor.get_action(policy)

            next_state: TimeStep = env.step(as_numpy(action))

            state, reward, done, _ = next_state, next_state.reward, next_state.step_type is StepType.LAST, "your mother"
            self.actor["episode_rewards"].append(reward)


            if step % 5000 or video_step_counter != 0:
                video_step_counter += 1
                frame = env.render()
                video_creator.append(frame)
                if video_step_counter == 1000:
                    print("Bitchass saving")
                    imageio.mimsave(f"videos/video_{step}.mp4", video_creator, fps=30)
                    video_creator = []
                    video_step_counter = 0




                

            if done:
                # Perform update at the end of each episode
                pi_loss, v_loss = self.update()
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())

                # Test policy
                render = episode_num % 50 == 0
                test_result = self.test(env, render=render)
                print("Bitch ass test result", test_result)
                if test_result > max_reward:
                    max_reward = test_result
                    model_path = (
                        self.model_dir +"/" + f"a2c-panda-{int(test_result)}-{step}.pt"
                    )
                    torch.save(self.state_dict(), model_path)
                state = env.reset()
                episode_num += 1

                self.save_reward_plot(plot_path = "reward_plot/rewards_plot.png" )


        model_path = (
            self.model_dir +"/" + f"a2c-panda-end.pt"
        )
        torch.save(self.state_dict(), model_path)

        self.save_reward_plot(plot_path = "reward_plot/rewards_plot.png" )
        return stats
    
    def save_reward_plot(self, plot_path):
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(self.accumulated_rewards), label="Rewards", color='blue', linestyle='-', marker='o', markersize=4)
        plt.title("Rewards Over Time", fontsize=16)
        plt.xlabel("Episodes", fontsize=14)
        plt.ylabel("Reward", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(plot_path)
        plt.show()
        

    def test(
        self, env, max_steps=25_000, test_iterations=10, render=False, sleep_time=None
    ):
        episode_rewards = []

        mse_threshold = 1e-4
        stationary_count_threshold = 50

        for episode in tqdm(range(test_iterations)):

            stationary_count = 0
            total_reward = 0
            state_curr = env.reset()
            for step in range(max_steps):
                _, policy = self(state_curr.observation)
                action, _ = self.actor.get_action(policy)


                next_state: TimeStep = env.step(as_numpy(action))

                state_next, reward, done, _ = next_state, next_state.reward, next_state.step_type is StepType.LAST, "your mother but in test"

                mse = ((state_curr.observation - state_next.observation) ** 2).sum()
                if mse < mse_threshold:
                    stationary_count += 1
                if stationary_count >= stationary_count_threshold:
                    done = True

                state_curr = state_next
                total_reward += reward

                if render:
                    env.render()
                    if sleep_time:
                        time.sleep(sleep_time)
                if done:
                    break
            episode_rewards.append(total_reward)
            render = False

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        self.accumulated_rewards.append(mean_reward)
        self.clear_buffers()
        return mean_reward

    def clear_buffers(self):
        self.actor.clear_buffers()
        self.critic.clear_buffers()


def test_policy(a2c, env, max_steps):
    observation = env.reset()
    for step in range(max_steps):
        env.render()
        value, policy = a2c(observation)
        action, logp_a = a2c.actor.get_action(policy)
        observation, _, done, _ = env.step(action.numpy())
        if done:
            break
    env.render()
    a2c.clear_buffers()

