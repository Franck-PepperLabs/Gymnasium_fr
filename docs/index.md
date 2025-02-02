---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/img/gymnasium-text.png
:alt: Logo de Gymnasium
```

```{project-heading}
Une API standard pour l'apprentissage par renforcement avec une collection diversifiée d'environnements de référence
```

```{figure} _static/videos/box2d/lunar_lander.gif
   :alt: Lunar Lander
   :width: 500
```

**Gymnasium est un fork maintenu de la bibliothèque Gym d'OpenAI.** L'interface de Gymnasium est simple, pythonique et capable de représenter des problèmes d'apprentissage par renforcement généraux, et dispose d'un [adaptateur de compatibilité](content/gym_compatibility) pour les anciens environnements de Gym :

```{code-block} python

import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # c'est ici que vous inséreriez votre politique
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
```

```{toctree}
:hidden:
:caption: Introduction

content/basic_usage_fr
content/gym_compatibility_fr
content/migration-guide_fr
```

```{toctree}
:hidden:
:caption: API

api/env_fr
api/registry
api/spaces
api/wrappers
api/vector
api/utils
api/experimental
```

```{toctree}
:hidden:
:caption: Environnements

environments/classic_control
environments/box2d
environments/toy_text
environments/mujoco
environments/atari
environments/third_party_environments
```

```{toctree}
:hidden:
:glob:
:caption: Tutoriels

tutorials/**/index
Tutoriel Comet <https://www.comet.com/docs/v2/integrations/ml-frameworks/gymnasium/?utm_source=gymnasium&utm_medium=partner&utm_campaign=partner_gymnasium_2023&utm_content=docs_gymnasium>
```

```{toctree}
:hidden:
:caption: Développement

Github <https://github.com/Farama-Foundation/Gymnasium>
gymnasium_release_notes/index
gym_release_notes/index
Contribuer à la documentation <https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/README.md>
```
