### Created using tensorflow's categorical DQN tutorial: https://www.tensorflow.org/agents/tutorials/9_c51_tutorial#setup

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import numpy as np
import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts
from tf_agents.replay_buffers import tf_uniform_replay_buffer

import matplotlib
import matplotlib.pyplot as plt



class Card:
    values = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    def __init__(self, v):
        self.value = v

class Deck:
    def __init__(self):
        from random import shuffle
        self.cards = []
        for _ in range(10):
            for i in range(2, 15):
                for _ in range(4):
                    self.cards.append(Card(i))
        shuffle(self.cards)
    def rm_card(self):
        if len(self.cards) == 0:
            self = Deck()
        return self.cards.pop()

class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self.deck = Deck()
        self.cards = []
        self._state = [0, 0, 0, 0] #[player card sum, dealer public card, number of aces, num of cards]
        self._doubled = 1
        self.rule_breaks = 0

        self.dealer_cards = []
        self.dealer_cards.append(self.deck.rm_card())
        self.dealer_turn = True
        self.dealer_sum = self.dealer_cards[0].value
        self._state[1] = self.dealer_sum

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,4), dtype=np.int32, minimum=0, name='observation')
        self._episode_ended = False 

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0, 0, 0, 0]
        self._episode_ended = False
        self.cards = []
        self._doubled = 1
        self.rule_breaks = 0
        for _ in range(2):
            new_card = self.deck.rm_card()
            self.cards.append(new_card)
            self._state[0] += new_card.value
            self._state[3] += 1
        self.playerAceSwap()

        self.dealer_cards = []
        self.dealer_cards.append(self.deck.rm_card())
        self.dealer_turn = True
        self.dealer_sum = self.dealer_cards[0].value
        self._state[1] = self.dealer_sum

        for card in self.cards:
            if card.value == 11: self._state[2] += 1

        return ts.restart(np.array([self._state], dtype=np.int32))

    def winnerLogic(self):
        if self._state[0] <= 21:
            if self._state[0] > self.dealer_sum or self.dealer_sum > 21:
                return True
        return False
    def playerAceSwap(self):
        if self._state[0] < 21: return
        self._state[0] = 0
        for ind in range(len(self.cards)):
            if self.cards[ind].value == 11:
                self.cards[ind].value == 1
            self._state[0] += self.cards[ind].value
    def dealerAceSwap(self):
        if self.dealer_sum < 21: return
        self.dealer_sum = 0
        for ind in range(len(self.dealer_cards)):
            if self.dealer_cards[ind].value == 11:
                self.dealer_cards[ind].value == 1
            self.dealer_sum += self.dealer_cards[ind].value
            

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        # TODO make it so that there is a limit for how many double downs/splits so that AI can't infinetly delay game.
        if action == 0:
            self._episode_ended = True
        elif action == 1: 
            new_card = self.deck.rm_card()
            self.cards.append(new_card)
            self._state[0] += new_card.value
            self._state[3] += 1
        elif action == 2:
            if len(self.cards) != 2:
                self.rule_breaks += 1
            else:
                new_card = self.deck.rm_card()
                self.cards.append(new_card)
                self._state[0] += new_card.value
                self._state[3] += 1
                self._episode_ended = True
                self._doubled = 2 
        elif action == 3:
            if len(self.cards) != 2 or self.cards[0] != self.cards[1]:
                self.rule_breaks += 1
            else:
                self.cards[1] = self.deck.rm_card()
                self._state[0] = 0
                for card in self.cards:
                    self._state[0] += card.value

        else:
            raise ValueError('`action` should be 0, 1, 2, or 3.')

        self.playerAceSwap()
        if self._state[0] >= 21:
            self._episode_ended = True
        elif self.rule_breaks > 10:
            self._episode_ended = True
        self._state[2] = 0
        for card in self.cards:
            if card.value == 11: self._state[2] += 1

        if self._episode_ended:
            self.dealer_turn = True
            while self.dealer_turn == True:
                self.dealer_cards.append(self.deck.rm_card())
                self.dealer_sum += self.dealer_cards[-1].value
                self.dealerAceSwap()
                if self.dealer_sum >= 17:
                    self.dealer_turn = False

        if self._episode_ended: 
            reward = 1 if self.winnerLogic() else -1
            return ts.termination(np.array([self._state], dtype=np.int32), reward*self._doubled)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)

train_Env_py = CardGameEnv()
eval_Env_py = CardGameEnv()
train_env = tf_py_environment.TFPyEnvironment(train_Env_py)
eval_env = tf_py_environment.TFPyEnvironment(eval_Env_py)

num_iterations = 100_000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_capacity = 10_000  # @param {type:"integer"}

fc_layer_params = (50, 150, 50,)

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
gamma = 0.99
log_interval = 1_000  # @param {type:"integer"}

num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}

num_eval_episodes = 500  # @param {type:"integer"}
eval_interval = 5000  # @param {type:"integer"}

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

def compute_avg_return(environment, policy, num_episodes=100):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(initial_collect_steps):
  collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience)

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
    returns.append(avg_return)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=550)