---
title: Utils
---

# Env

## gymnasium.Env

```{eval-rst}
.. autoclass:: gymnasium.Env
```

### Méthodes

```{eval-rst}
.. autofunction:: gymnasium.Env.step
.. autofunction:: gymnasium.Env.reset
.. autofunction:: gymnasium.Env.render
```

### Attributs

```{eval-rst}
.. autoattribute:: gymnasium.Env.action_space

    L'objet Space correspondant aux actions valides, toutes les actions valides doivent être contenues dans cet espace. Par exemple, si l'espace d'action est de type `Discrete` et qu'il donne la valeur `Discrete(2)`, cela signifie qu'il y a deux actions discrètes valides : 0 et 1.

    .. code::

        >>> env.action_space
        Discrete(2)
        >>> env.observation_space
        Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)

.. autoattribute:: gymnasium.Env.observation_space

    L'objet Space correspondant aux observations valides, toutes les observations valides doivent être contenues dans cet espace. Par exemple, si l'espace d'observation est de type :class:`Box` et que la forme de l'objet est ``(4,)``, cela signifie qu'une observation valide sera un tableau de 4 nombres. Nous pouvons également vérifier les limites de la boîte avec les attributs.

    .. code::

        >>> env.observation_space.high
        array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38], dtype=float32)
        >>> env.observation_space.low
        array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38], dtype=float32)

.. autoattribute:: gymnasium.Env.metadata

    Les métadonnées de l'environnement contenant les modes de rendu, les images par seconde du rendu, etc.

.. autoattribute:: gymnasium.Env.render_mode

    Le mode de rendu de l'environnement déterminé lors de l'initialisation.

.. autoattribute:: gymnasium.Env.reward_range

    Un tuple correspondant aux récompenses minimale et maximale possibles pour un agent au cours d'un épisode. La plage de récompenses par défaut est définie à :math:`(-\infty,+\infty)`.

.. autoattribute:: gymnasium.Env.spec

    Le ``EnvSpec`` de l'environnement normalement défini lors de l'appel à :py:meth:`gymnasium.make`
```

### Méthodes supplémentaires

```{eval-rst}
.. autofunction:: gymnasium.Env.close
.. autoproperty:: gymnasium.Env.unwrapped
.. autoproperty:: gymnasium.Env.np_random
```

### Implémentation des environnements

```{eval-rst}
.. py:currentmodule:: gymnasium

Lors de l'implémentation d'un environnement, les fonctions :meth:`Env.reset` et :meth:`Env.step` doivent être créées pour décrire la dynamique de l'environnement.
Pour plus d'informations, consultez le tutoriel sur la création d'environnements.
```