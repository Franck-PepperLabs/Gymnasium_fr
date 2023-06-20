[![pre-commit](https://img.shields.io/badge/pre--commit-activé-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/style%20de%20code-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="500px"/>
</p>

Gymnasium est une bibliothèque Python open source permettant de développer et de comparer des algorithmes d'apprentissage par renforcement en fournissant une API standard pour communiquer entre les algorithmes d'apprentissage et les environnements, ainsi qu'un ensemble standard d'environnements conformes à cette API. Il s'agit d'une dérivation de la bibliothèque [Gym](https://github.com/openai/gym) d'OpenAI par ses mainteneurs (OpenAI a confié la maintenance à une équipe externe il y a quelques années), et c'est là que se déroulera la maintenance future.

Le site de documentation se trouve à l'adresse [gymnasium.farama.org](https://gymnasium.farama.org), et nous avons un serveur Discord public (que nous utilisons également pour coordonner les travaux de développement) auquel vous pouvez vous joindre ici : https://discord.gg/bnJ6kubTg6

## Environnements

Gymnasium comprend les familles d'environnements suivantes, ainsi qu'une grande variété d'environnements tiers :
* [Contrôle classique](https://gymnasium.farama.org/fr/environments/classic_control/) - Il s'agit d'environnements classiques d'apprentissage par renforcement basés sur des problèmes du monde réel et de la physique.
* [Box2D](https://gymnasium.farama.org/fr/environments/box2d/) - Ces environnements sont tous des jeux jouets basés sur le contrôle physique, utilisant la physique basée sur Box2D et le rendu basé sur PyGame.
* [Toy Text](https://gymnasium.farama.org/fr/environments/toy_text/) - Ces environnements sont conçus pour être extrêmement simples, avec de petits espaces d'états et d'actions discrets, et donc faciles à apprendre. Par conséquent, ils conviennent au débogage des implémentations d'algorithmes d'apprentissage par renforcement.
* [MuJoCo](https://gymnasium.farama.org/fr/environments/mujoco/) - Des environnements basés sur un moteur physique avec un contrôle multi-articulaire qui sont plus complexes que les environnements basés sur Box2D.
* [Atari](https://gymnasium.farama.org/fr/environments/atari/) - Un ensemble de 57 environnements Atari 2600 simulés à l'aide de Stella et de l'Arcade Learning Environment, offrant une grande complexité pour les agents à apprendre.
* [Tiers](https://gymnasium.farama.org/fr/environments/third_party_environments/) - Plusieurs environnements compatibles avec l'API de Gymnasium ont été créés. Veuillez prendre en compte la version pour laquelle le logiciel a été créé et utilisez la fonction `apply_env_compatibility` dans `gymnasium.make` si nécessaire.

## Installation

Pour installer la bibliothèque de base Gymnasium, utilisez la commande `pip install gymnasium`

Cela n'inclut pas les dépendances pour toutes les familles d'environnements (il y en a un grand nombre, et certaines peuvent poser des problèmes lors de l'installation sur certains systèmes). Vous pouvez installer ces dépendances pour une famille spécifique avec `pip install "gymnasium[atari]"`, ou utiliser `pip install "gymnasium[all]"` pour installer toutes les dépendances.

Nous prenons en charge et testons Python 3.7, 3.8, 3.9, 3.10, 3.11 sur Linux et macOS. Nous accepterons les demandes de pull liées à Windows, mais nous ne le prenons pas en charge officiellement.

## API

L'API de Gymnasium modélise les environnements en tant que simples classes Python `env`. La création d'instances d'environnement et l'interaction avec eux est très simple. Voici un exemple utilisant l'environnement "CartPole-v1" :

```python
import gymnasium as gym
env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

## Bibliothèques connexes notables

Veuillez noter qu'il s'agit d'une liste incomplète, qui inclut les bibliothèques auxquelles les mainteneurs renvoient le plus souvent les nouveaux arrivants lorsqu'ils demandent des recommandations.

* [CleanRL](https://github.com/vwxyzjn/cleanrl) est une bibliothèque d'apprentissage basée sur l'API de Gymnasium. Elle est conçue pour les personnes débutantes dans le domaine et propose des implémentations de référence très performantes.
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) est une version multi-agent de Gymnasium avec plusieurs environnements implémentés, notamment des environnements Atari multi-agents.
* La Fondation Farama propose également une collection de nombreux autres [environnements](https://farama.org/projects) entretenus par la même équipe que Gymnasium et utilisant l'API de Gymnasium.
* [Comet](https://www.comet.com/site/?utm_source=gymnasium&utm_medium=partner&utm_campaign=partner_gymnasium_2023&utm_content=github) est un outil gratuit de ML-Ops qui suit les récompenses, les métriques, les hyperparamètres et le code des exécutions d'apprentissage automatique. Comet s'intègre facilement à Gymnasium, voici un [tutoriel](https://bit.ly/CometGymnasiumIntegration) sur l'utilisation des deux ensemble ! Comet est un sponsor de la Fondation Farama !

## Versionnement des environnements

Gymnasium respecte une politique de versionnement strict pour des raisons de reproductibilité. Tous les environnements se terminent par un suffixe tel que "-v0". Lorsque des modifications sont apportées aux environnements pouvant avoir un impact sur les résultats d'apprentissage, le nombre est augmenté de un pour éviter toute confusion potentielle. Cette pratique provient de Gym.

## Feuille de route de développement

Nous avons une feuille de route pour le développement futur de Gymnasium disponible ici : https://github.com/Farama-Foundation/Gymnasium/issues/12

## Soutenez le développement de Gymnasium

Si vous avez les moyens financiers de le faire et que vous souhaitez soutenir le développement de Gymnasium, rejoignez d'autres membres de la communauté en [nous faisant un don](https://github.com/sponsors/Farama-Foundation).