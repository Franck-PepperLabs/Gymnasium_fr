# fmt: off
"""
Entraînement à l'aide de REINFORCE pour Mujoco
=============================================

.. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig1.gif
  :width: 400
  :alt: agent-environment-diagram

Ce tutoriel sert à 2 fins :
 1. Comprendre comment implémenter REINFORCE [1] à partir de zéro pour résoudre InvertedPendulum-v4 de Mujoco.
 2. Implémenter un algorithme d'apprentissage par renforcement profond avec la fonction `step()` de Gymnasium v0.26+.

Nous utiliserons **REINFORCE**, l'une des premières méthodes de gradient de politique. Contrairement à l'approche consistant à apprendre d'abord une fonction de valeur, puis à en déduire une politique,
REINFORCE optimise directement la politique. En d'autres termes, il est entraîné pour maximiser la probabilité des retours de Monte-Carlo. Plus d'informations à ce sujet seront données ultérieurement.

**Inverted Pendulum** est le cartpole de Mujoco, mais maintenant alimenté par le simulateur physique de Mujoco,
ce qui permet des expériences plus complexes (comme varier les effets de la gravité).
Cet environnement implique un chariot qui peut être déplacé linéairement, avec un poteau fixé à une extrémité et ayant une autre extrémité libre.
Le chariot peut être poussé vers la gauche ou la droite, et le but est de maintenir le poteau en équilibre au sommet du chariot en appliquant des forces sur le chariot.
Plus d'informations sur l'environnement peuvent être trouvées sur https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/

**Objectifs de l'entraînement** : Maintenir le poteau (pendule inversé) en équilibre au sommet du chariot.

**Actions** : L'agent prend un vecteur 1D pour les actions. L'espace des actions est continu ``(action)`` dans ``[-3, 3]``,
où l'action représente la force numérique appliquée au chariot
(la magnitude représente l'intensité de la force et le signe représente la direction).

**Approche** : Nous utilisons PyTorch pour coder REINFORCE à partir de zéro afin d'entraîner une politique de réseau neuronal pour maîtriser le pendule inversé.

Explication de la fonction `Env.step()` de Gymnasium v0.26+

``env.step(A)`` nous permet d'effectuer une action 'A' dans l'environnement actuel 'env'. L'environnement exécute alors l'action
et renvoie cinq variables :

-  ``next_obs`` : Il s'agit de l'observation que l'agent recevra après avoir effectué l'action.
-  ``reward`` : Il s'agit de la récompense que l'agent recevra après avoir effectué l'action.
-  ``terminated`` : Il s'agit d'une variable booléenne qui indique si l'environnement s'est terminé ou non.
-  ``truncated`` : Il s'agit également d'une variable booléenne qui indique si l'épisode s'est terminé par une troncature anticipée, c'est-à-dire si une limite de temps est atteinte.
-  ``info`` : Il s'agit d'un dictionnaire qui peut contenir des informations supplémentaires sur l'environnement.
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)


# %%
# Réseau de politique
# ~~~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig2.png
#
# Nous commençons par construire une politique que l'agent apprendra en utilisant REINFORCE.
# Une politique est une correspondance entre l'observation actuelle de l'environnement et une distribution de probabilité des actions à prendre.
# La politique utilisée dans le tutoriel est paramétrée par un réseau neuronal. Elle est composée de 2 couches linéaires partagées à la fois pour la moyenne prédite et l'écart type.
# De plus, des couches linéaires individuelles distinctes sont utilisées pour estimer la moyenne et l'écart type. La non-linéarité ``nn.Tanh`` est utilisée entre les couches cachées.
# La fonction suivante estime une moyenne et un écart type d'une distribution normale à partir de laquelle une action est échantillonnée. Ainsi, il est prévu que la politique apprenne
# les poids appropriés pour produire des moyennes et des écarts types en fonction de l'observation actuelle.


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs


# %%
# Construction d'un agent
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig3.jpeg
#
# Maintenant que nous avons terminé de construire la politique, développons **REINFORCE** qui donne vie au réseau de politique.
# L'algorithme REINFORCE peut être trouvé ci-dessus. Comme mentionné précédemment, l'objectif de REINFORCE est de maximiser les rendements de Monte-Carlo.
#
# Petit fait amusant : REINFROCE est un acronyme pour " 'RE'ward 'I'ncrement 'N'on-negative 'F'actor times 'O'ffset 'R'einforcement times 'C'haracteristic 'E'ligibility".
#
# Remarque : Le choix des hyperparamètres vise à entraîner un agent qui se comporte de manière décente. Aucun réglage intensif des hyperparamètres n'a été effectué.
#


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []


# %%
# Entraînons maintenant la politique en utilisant REINFORCE pour maîtriser la tâche du pendule inversé.
#
# Voici un aperçu de la procédure d'entraînement :
#
#    pour chaque graine dans les graines aléatoires
#        réinitialiser l'agent
#
#        pour épisode dans la plage du nombre maximal d'épisodes
#            jusqu'à ce que l'épisode soit terminé
#                échantillonner une action basée sur l'observation actuelle
#
#                effectuer l'action, recevoir la récompense et l'observation suivante
#
#                stocker l'action effectuée, sa probabilité et la récompense observée
#            mettre à jour la politique
#
# Remarque : L'apprentissage par renforcement profond est assez sensible à la graine aléatoire dans de nombreux cas d'utilisation courants (https://spinningup.openai.com/en/latest/spinningup/spinningup.html).
# Il est donc important de tester différentes graines, ce que nous ferons.


# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)


# %%
# Tracer la courbe d'apprentissage
# ~~~~~~~~~~~~~~~~~~~


rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()


# %%
# .. image:: /_static/img/tutorials/reinforce_invpend_gym_v26_fig4.png
#
# Auteur: Siddarth Chandrasekar
#
# Licence: Licence MIT
#
# Références
# ~~~~~~~~~~
#
# [1] Williams, Ronald J.. "Simple statistical gradient-following
# algorithms for connectionist reinforcement learning." Machine Learning 8
# (2004): 229-256.
#
