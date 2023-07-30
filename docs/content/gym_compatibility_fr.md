---
layout: "contents"
title: Compatibilité avec Gym
---

# Compatibilité avec Gym

Gymnasium offre plusieurs méthodes de compatibilité pour une gamme d'implémentations d'environnements.

## Chargement des environnements OpenAI Gym

```{eval-rst}
.. py:currentmodule:: gymnasium.wrappers

Pour les environnements enregistrés uniquement dans OpenAI Gym et non dans Gymnasium, Gymnasium v0.26.3 et versions ultérieures permettent de les importer soit via un environnement spécial, soit via un adaptateur.
L'environnement ``"GymV26Environment-v0"`` a été introduit dans Gymnasium v0.26.3 et permet d'importer les environnements Gym via l'argument ``env_name`` ainsi que d'autres arguments pertinents de l'environnement.
Pour effectuer une conversion via un adaptateur, l'environnement lui-même peut être passé à l'adaptateur :class:`EnvCompatibility` via l'argument ``env``.
```

Un exemple en est atari 0.8.0 qui n'a pas d'implémentation dans Gymnasium.
```python
import gymnasium as gym

env = gym.make("GymV26Environment-v0", env_id="ALE/Pong-v5")
```

## Compatibilité avec l'environnement Gym v0.21

```{eval-rst}
.. py:currentmodule:: gymnasium

Plusieurs environnements n'ont pas été mis à jour avec les modifications récentes de Gym, en particulier depuis la version v0.21.
Cette mise à jour est significative pour l'introduction des signatures ``termination`` et ``truncation`` en remplacement de l'ancienne signature ``done`` utilisée précédemment.
Pour assurer la compatibilité ascendante, Gym et Gymnasium v0.26+ incluent un argument ``apply_api_compatibility`` lors de l'appel de la méthode :meth:`make` qui convertit automatiquement un environnement conforme à l'API v0.21 en un environnement compatible avec v0.26+.
```

```python
import gym

env = gym.make("OldV21Env-v0", apply_api_compatibility=True)
```

De plus, dans Gymnasium, nous fournissons des environnements spécialisés pour la compatibilité qui, pour ``env_id``, appelleront ``gym.make``.
```python
import gymnasium

env = gymnasium.make("GymV21Environment-v0", env_id="CartPole-v1", render_mode="human")
# ou
env = gymnasium.make("GymV21Environment-v0", env=OldV21Env())

```

## Compatibilité de l'API Step

```{eval-rst}
Si les environnements implémentent l'ancienne API du pas (step) avec l'attribut ``done``, Gymnasium fournit à la fois des fonctions (:meth:`gymnasium.utils.step_api_compatibility.convert_to_terminated_truncated_step_api`) et des adaptateurs (:class:`gymnasium.wrappers.StepAPICompatibility`) qui convertiront un environnement utilisant l'ancienne API du pas (step) (avec ``done``) vers la nouvelle API (avec ``termination`` et ``truncation``).
```
