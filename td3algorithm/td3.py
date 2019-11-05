import copy

from actor import Actor
from critic import Critic
import numpy as np
import torch
import torch.nn.functional as F


class TD3(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.

    Args:
        state_dim (array): state size
        action_dim (array): action size
        policy_noise (float): how much noise to add to actions
        device (device): cuda or cpu to process the tensors
        discount (float): discount factor
        tau (float): soft update for main networks to target networks
        policy_noise (float): noise factor
        noise_clip (float): clip factor
        policy_freq (int): frequency of policy updates

    """

    def __init__(self, state_dim, action_dim, max_action, discount, tau, policy_noise, noise_clip, policy_freq, device):

        self.state_dim = len(state_dim[0])
        self.action_dim = len(action_dim)
        self.max_action = max_action[2]
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).float()
        # self.actor_target = Actor(state_dim, action_dim, self.max_action).to(device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # or 1e-3

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).float()
        # self.critic_target = Critic(state_dim, action_dim).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)  # or 1e-2

        self.device = device
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        """Select an appropriate action from the agent policy
            Args:
                state (array): current state of environment

            Returns:
                action (float): action clipped within action range
        """

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        #  if noise != 0:
        #      action_dim = len(self.env.action_space())
        #      action = (action + np.random.normal(0, noise, size=action_dim))
        #  action_space_low, _, action_space_high = self.env.action_domain()
        #  return action.clip(action_space_low, action_space_high)

        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        """Train and update actor and critic networks
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                batch_size(int): batch size to sample from replay buffer
            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network
        """
        self.total_it += 1
        # Sample replay buffer
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)

        state = torch.tensor(np.array([np.array(i.item().values()) for i in state]))
        next_state = np.asarray([np.array(i.item().values()) for i in next_state])
        reward = torch.as_tensor(reward, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)

        with torch.no_grad():
            # select an action according to the policy an add clipped noise
            # need to select set of actions
            noise = (torch.rand_like(torch.from_numpy(action)) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(torch.tensor(next_state, dtype=torch.float32)) +
                           torch.tensor(noise, dtype=torch.float32)).clamp(self.max_action[0], self.max_action[2])
            # next_action_d =torch.as_tensor(next_action, dtype=torch.double)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic(state, next_action)
            target_Q = torch. min(target_Q1, target_Q2)
            target_Q = reward + done * self.discount * target_Q

        # update action datatype, can't do earlier, use np.array earlier
        action = torch.as_tensor(action, dtype=torch.float32)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # compute the actor loss
            actor_loss = -self.critic.get_q(state, self.actor(state)).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
