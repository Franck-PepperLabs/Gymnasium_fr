---
layout: "contents"
title: Utilisation basique
firstpage:
---

# Utilisation basique

Gymnasium est un projet qui fournit une API pour tous les environnements d'apprentissage par renforcement à agent unique et inclut des implémentations d'environnements courants tels que CartPole, Pendulum, MountainCar, Mujoco, Atari, et plus encore.

L'API contient quatre fonctions clés : ``make``, ``reset``, ``step`` et ``render``, que cette utilisation basique vous présentera. Au cœur de Gymnasium se trouve ``Env``, une classe Python de haut niveau représentant un processus de décision markovien (MDP) de la théorie de l'apprentissage par renforcement (ceci n'est pas une reconstruction parfaite et il manque plusieurs composants des MDP). Dans Gymnasium, les environnements (MDP) sont implémentés sous forme de classes ``Env``, ainsi que de ``Wrappers``, qui fournissent des utilitaires pratiques et peuvent modifier les résultats transmis à l'utilisateur.

## Initialisation des environnements

L'initialisation des environnements est très facile dans Gymnasium et peut être effectuée via la fonction ``make`` :

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
```

Cela renverra un objet ``Env`` avec lequel les utilisateurs peuvent interagir. Pour voir tous les environnements que vous pouvez créer, utilisez ``gymnasium.envs.registry.keys()``. ``make`` inclut un certain nombre de paramètres supplémentaires pour ajouter des wrappers, spécifier des mots clés à l'environnement, et plus encore.

## Interaction avec l'environnement

La boucle classique "agent-environnement" illustrée ci-dessous est une représentation simplifiée de l'apprentissage par renforcement que Gymnasium met en œuvre.

```{image} /_static/diagrams/AE_loop.png
:width: 50%
:align: center
:class: only-light
```

```{image} /_static/diagrams/AE_loop_dark.png
:width: 50%
:align: center
:class: only-dark
```

Cette boucle est implémentée dans le code Gymnasium suivant :

```python
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # politique de l'agent qui utilise l'observation et l'information
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

La sortie devrait ressembler à ceci :

```{figure} https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif
:width: 50%
:align: center
```

### Explication du code

Tout d'abord, un environnement est créé à l'aide de ``make`` avec un mot clé supplémentaire ``"render_mode"`` qui spécifie comment l'environnement doit être visualisé. Consultez la fonction ``render`` pour plus de détails sur la signification par défaut des différents modes de rendu. Dans cet exemple, nous utilisons l'environnement ``"LunarLander"`` où l'agent contrôle un vaisseau spatial qui doit atterrir en toute sécurité.

Après avoir initialisé l'environnement, nous utilisons la fonction ``reset`` pour obtenir la première observation de l'environnement. Pour initialiser l'environnement avec une graine aléatoire particulière ou des options (consultez la documentation de l'environnement pour les valeurs possibles), utilisez les paramètres ``seed`` ou ``options`` avec la fonction ``reset``.

Ensuite, l'agent effectue une action dans l'environnement avec la fonction ``step``. On peut imaginer cela comme le déplacement d'un robot ou l'appui sur un bouton d'une manette de jeu qui provoque un changement dans l'environnement. En conséquence, l'agent reçoit une nouvelle observation de l'environnement mis à jour ainsi qu'une récompense pour avoir effectué l'action. Cette récompense peut être positive, par exemple pour avoir détruit un ennemi, ou une récompense négative, par exemple pour s'être déplacé dans de la lave. Un tel échange action-observation est appelé *pas de temps*.

Cependant, après quelques pas de temps, l'environnement peut se terminer, c'est ce qu'on appelle l'état terminal. Par exemple, le robot peut s'écraser ou l'agent peut avoir réussi à accomplir une tâche, l'environnement doit s'arrêter car l'agent ne peut pas continuer. Dans Gymnasium, si l'environnement est terminé, cela est renvoyé par la fonction ``step``. De même, nous pouvons également vouloir que l'environnement se termine après un nombre fixe de pas de temps, dans ce cas, l'environnement émet un signal tronqué. Si l'une des conditions ``terminated`` ou ``truncated`` est `True`, la fonction ``reset`` doit être appelée ensuite pour redémarrer l'environnement.

## Espaces d'action et d'observation

Chaque environnement spécifie le format des actions valides et des observations avec les attributs ``env.action_space`` et ``env.observation_space``. Cela est utile pour connaître à la fois l'entrée et la sortie attendues de l'environnement, car toutes les actions et observations valides doivent être contenues dans l'espace correspondant.

Dans l'exemple, nous avons échantillonné des actions aléatoires avec ``env.action_space.sample()`` au lieu d'utiliser une politique d'agent qui mappe les observations aux actions que les utilisateurs voudront effectuer. Consultez l'un des tutoriels sur les agents pour un exemple de création et d'entraînement d'une politique d'agent.

Chaque environnement devrait avoir les attributs ``action_space`` et ``observation_space``, tous deux étant des instances de classes héritées de ``Space``. Gymnasium prend en charge la plupart des espaces possibles dont les utilisateurs pourraient avoir besoin :

- ``Box`` : décrit un espace continu de dimension n. Il s'agit d'un espace borné dans lequel nous pouvons définir les limites supérieure et inférieure qui décrivent les valeurs valides que peuvent prendre nos observations.
- ``Discrete`` : décrit un espace discret où {0, 1, ..., n-1} sont les valeurs possibles que peut prendre notre observation ou notre action. Les valeurs peuvent être décalées en utilisant un argument facultatif pour obtenir {a, a+1, ..., a+n-1}.
- ``Dict`` : représente un dictionnaire d'espaces simples.
- ``Tuple`` : représente un tuple d'espaces simples.
- ``MultiBinary`` : crée un espace binaire de forme n. L'argument n peut être un nombre ou une liste de nombres.
- ``MultiDiscrete`` : se compose d'une série d'espaces d'actions ``Discrete`` avec un nombre d'actions différent pour chaque élément.

Pour des exemples d'utilisation des espaces, consultez leur [documentation](/api/spaces) ainsi que les [fonctions utilitaires](/api/spaces/utils). Il existe également quelques espaces plus spécialisés tels que ``Graph``, ``Sequence`` et ``Text``.

## Modification de l'environnement

Les wrappers sont un moyen pratique de modifier un environnement existant sans avoir à modifier directement le code sous-jacent. L'utilisation de wrappers vous permet d'éviter une grande partie du code redondant et de rendre votre environnement plus modulaire. Les wrappers peuvent également être chaînés pour combiner leurs effets. La plupart des environnements générés via ``gymnasium.make`` seront déjà enveloppés par défaut avec les wrappers ``TimeLimit``, ``OrderEnforcing`` et ``PassiveEnvChecker``.

Pour envelopper un environnement, vous devez d'abord initialiser un environnement de base. Ensuite, vous pouvez transmettre cet environnement ainsi que des paramètres (éventuellement facultatifs) au constructeur du wrapper :

```python
>>> import gymnasium as gym
>>> from gymnasium.wrappers import FlattenObservation
>>> env = gym.make("CarRacing-v2")
>>> env.observation_space.shape
(96, 96, 3)
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space.shape
(27648,)

```

Gymnasium fournit déjà de nombreux wrappers couramment utilisés. Voici quelques exemples :

- `TimeLimit` : émet un signal tronqué si un nombre maximal de pas de temps est dépassé (ou si l'environnement de base a émis un signal tronqué).
- `ClipAction` : limite l'action de sorte qu'elle soit dans l'espace des actions (de type ``Box``).
- `RescaleAction` : redimensionne les actions pour qu'elles se situent dans un intervalle spécifié.
- `TimeAwareObservation` : ajoute des informations sur l'indice du pas de temps à l'observation. Dans certains cas, cela peut être utile pour s'assurer que les transitions sont de type Markov.

Pour obtenir une liste complète des wrappers implémentés dans Gymnasium, consultez [wrappers](/api/wrappers).

Si vous avez un environnement enveloppé et que vous souhaitez obtenir l'environnement de base sous tous les niveaux de wrappers (afin de pouvoir appeler manuellement une fonction ou modifier un aspect sous-jacent de l'environnement), vous pouvez utiliser l'attribut `.unwrapped`. Si l'environnement est déjà un environnement de base, l'attribut `.unwrapped` renverra simplement l'environnement lui-même.

```python
>>> wrapped_env
<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v2>>>>>>
>>> wrapped_env.unwrapped
<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>

```

## Innformations complémentaires

* [Mettre en place un environnement personnalisé à l'aide de l'API Gymnasium](/tutorials/gymnasium_basics/environment_creation/)
* [Entraîner un agent à jouer au blackjack](/tutorials/training_agents/blackjack_tutorial)
* [Compatibilité avec OpenAI Gym](/content/gym_compatibility)
