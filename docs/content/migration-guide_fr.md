---
layout: "contents"
title: Guide de migration
---

# Guide de migration de v21 à v26

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Gymnasium est un fork de `OpenAI Gym v26 <https://github.com/openai/gym/releases/tag/0.26.2>`_, qui a introduit un changement majeur par rapport à `Gym v21 <https://github.com/openai/gym/releases/tag/v0.21.0>`_.
Dans ce guide, nous présentons brièvement les changements d'API de Gym v21 - pour lesquels plusieurs tutoriels ont été rédigés - à Gym v26.
Pour les environnements encore bloqués dans l'API v21, les utilisateurs peuvent utiliser l'adaptateur :class:`EnvCompatibility` pour les convertir en version compatible avec la v26.
Pour plus d'informations, consultez le `guide </content/gym_compatibility>`_
```

### Code d'exemple pour v21

```python
import gym
env = gym.make("LunarLander-v2", options={})
env.seed(123)
observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # politique de l'agent utilisant l'observation et info
    observation, reward, done, info = env.step(action)

    env.render(mode="human")

env.close()
```

### Code d'exemple pour v26

```python
import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=123, options={})

done = False
while not done:
    action = env.action_space.sample()  # politique de l'agent utilisant l'observation et info
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
```

## Seed et générateur de nombres aléatoires

```{eval-rst}
.. py:currentmodule:: gymnasium.Env

La méthode ``Env.seed()`` a été supprimée des environnements Gym v26 au profit de ``Env.reset(seed=seed)``.
Cela permet de changer la graine (seed) uniquement lors de la réinitialisation de l'environnement.
La décision de supprimer ``seed`` a été prise car certains environnements utilisent des émulateurs qui ne peuvent pas modifier les générateurs de nombres aléatoires au sein d'un épisode et doivent le faire au début d'un nouvel épisode.
Nous sommes conscients des cas où le contrôle du générateur de nombres aléatoires est important. Dans ces cas, si l'environnement utilise le générateur de nombres aléatoires intégré, les utilisateurs peuvent définir manuellement la graine avec l'attribut :attr:`np_random`.

Gymnasium v26 a changé pour utiliser ``numpy.random.Generator`` au lieu d'un générateur de nombres aléatoires personnalisé.
Cela signifie que plusieurs fonctions telles que ``randint`` ont été supprimées au profit de ``integers``.
Bien que certains environnements puissent utiliser un générateur de nombres aléatoires externe, nous recommandons d'utiliser l'attribut :attr:`np_random` auquel les adaptateurs et les utilisateurs externes peuvent accéder et utiliser.
```

## Réinitialisation de l'environnement

```{eval-rst}
Dans la v26, la méthode :meth:`reset` prend deux paramètres optionnels et renvoie une valeur.
Cela contraste avec la v21 qui ne prend pas de paramètres et renvoie ``None``.
Les deux paramètres sont ``seed`` pour définir le générateur de nombres aléatoires et ``options`` qui permet de transmettre des données supplémentaires à l'environnement lors de la réinitialisation.
Par exemple, dans le contrôle classique, le paramètre ``options`` permet désormais aux utilisateurs de modifier la plage de la limite d'état.
Consultez le `PR original <https://github.com/openai/gym/pull/2921>`_ pour plus de détails.

La méthode :meth:`reset` renvoie également ``info``, similaire à l'``info`` renvoyée par :meth:`step`.
Ceci est important car ``info`` peut inclure des métriques ou un masque d'actions valide qui est utilisé ou enregistré à l'étape suivante.

Pour mettre à jour les anciens environnements, nous recommandons fortement d'appeler ``super().reset(seed=seed)`` à la première ligne de :meth:`reset`.
Cela mettra automatiquement à jour :attr:`np_random` avec la valeur de la graine.
```

## Pas de l'environnement (Environment Step)

```{eval-rst}
Dans la v21, la définition de type de :meth:`step` est ``tuple[ObsType, SupportsFloat, bool, dict[str, Any]]`` représentant l'observation suivante, la récompense de l'étape, si l'épisode est terminé et des informations supplémentaires de l'étape.
En raison de problèmes de reproductibilité qui seront développés dans un prochain article de blog, nous avons modifié la définition de type en ``tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`` en ajoutant une valeur booléenne supplémentaire.
Ce booléen supplémentaire correspond à l'ancienne valeur ``done`` qui a été remplacée par ``terminated`` et ``truncated``.
Ces changements ont été introduits dans Gym `v26 <https://github.com/openai/gym/releases/tag/0.26.0>`_ (désactivés par défaut dans `v25 <https://github.com/openai/gym/releases/tag/0.25.0>`_).

Pour les utilisateurs souhaitant mettre à jour, dans la plupart des cas, remplacer ``done`` par ``terminated`` et ``truncated=False`` dans :meth:`step` devrait résoudre la plupart des problèmes.
Cependant, les environnements qui ont des raisons de tronquer plutôt que de terminer un épisode doivent lire le `PR associé <https://github.com/openai/gym/pull/2752>`_.
Pour les utilisateurs qui effectuent une boucle à travers un environnement, ils doivent modifier ``done = terminated or truncated`` comme indiqué dans le code d'exemple.
Pour les bibliothèques d'apprentissage, la différence principale est de changer ``done`` en ``terminated``, indiquant si un amortissement doit être effectué ou non.
```

## Wrapper TimeLimit

```{eval-rst}
Dans la v21, l'adaptateur :class:`TimeLimit` ajoutait une clé supplémentaire dans le dictionnaire ``info`` ``TimeLimit.truncated`` lorsque l'agent atteignait la limite de temps sans atteindre un état terminal.

Dans la v26, cette information est désormais communiquée via la valeur de retour `truncated` décrite dans la section précédente, qui vaut `True` si l'agent atteint la limite de temps, qu'il atteigne ou non un état terminal. L'ancienne entrée du dictionnaire est équivalente à ``truncated and not terminated``.
```

## Rendu de l'environnement (Environment Render)

```{eval-rst}
Dans la v26, une nouvelle API de rendu a été introduite, de sorte que le mode de rendu est fixé lors de l'initialisation car certains environnements n'autorisent pas les changements de mode de rendu en cours d'exécution. Par conséquent, les utilisateurs doivent désormais spécifier le :attr:`render_mode` dans ``gym.make``, comme indiqué dans le code d'exemple de la v26 ci-dessus.

Pour une explication plus complète des changements, veuillez consulter ce `résumé <https://younis.dev/blog/render-api/>`_.
```

## Code supprimé

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

* GoalEnv - Cela a été supprimé, les utilisateurs qui en ont besoin doivent réimplémenter l'environnement ou utiliser Gymnasium Robotics qui contient une implémentation de cet environnement.
* ``from gym.envs.classic_control import rendering`` - Cela a été supprimé au profit de la mise en œuvre de leurs propres systèmes de rendu. Les environnements Gymnasium sont codés en utilisant pygame.
* Environnements de robotique - Les environnements de robotique ont été déplacés vers le projet `Gymnasium Robotics <https://robotics.farama.org/>`_.
* Wrapper Monitor - Ce wrapper a été remplacé par deux wrappers séparés : :class:`RecordVideo` et :class:`RecordEpisodeStatistics`.

```
