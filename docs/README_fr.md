# Gymnasium-docs

Ce dossier contient la documentation pour [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

Si vous modifiez une page d'environnement Atari, veuillez suivre les instructions ci-dessous. Pour plus d'informations sur la contribution à la documentation, consultez notre fichier [CONTRIBUTING.md](https://github.com/Farama-Foundation/Celshast/blob/main/CONTRIBUTING.md).

## Instructions pour modifier les pages d'environnement

### Modification d'une page d'environnement

Si vous modifiez un environnement Atari, modifiez directement le fichier Markdown dans ce dépôt.

Sinon, effectuez un fork de Gymnasium et modifiez la chaîne de documentation dans le fichier Python de l'environnement. Ensuite, installez votre fork de Gymnasium via pip et exécutez `docs/_scripts/gen_mds.py` dans ce référentiel. Cela générera automatiquement un fichier de documentation Markdown pour l'environnement.

### Ajout d'un nouvel environnement

#### Environnement Atari

Pour les environnements Atari, ajoutez un fichier Markdown dans `docs/environments/atari`, puis effectuez les [autres étapes](#other-steps).

#### Environnement non-Atari

Assurez-vous que l'environnement est présent dans Gymnasium (ou dans votre fork). Vérifiez que le fichier Python de l'environnement possède une chaîne de documentation au format Markdown correctement formatée. Installez-le en utilisant `pip install -e .`, puis exécutez `docs/_scripts/gen_mds.py`. Cela générera automatiquement une page md pour l'environnement. Ensuite, effectuez les [autres étapes](#other-steps).

#### Autres étapes

- Ajoutez le GIF correspondant dans le dossier `docs/_static/videos/{ENV_TYPE}`, où `ENV_TYPE` est la catégorie de votre nouvel environnement (par exemple, mujoco). Suivez la convention de nommage en snake_case. Sinon, exécutez `docs/_scripts/gen_gifs.py`.
- Modifiez `docs/environments/{ENV_TYPE}/index.md` et ajoutez le nom du fichier correspondant à votre nouvel environnement à la table des matières (`toctree`).

## Générer la documentation

Installez les packages requis ainsi que Gymnasium (ou votre fork) :

```
pip install gymnasium
pip install -r docs/requirements.txt
```

Pour générer la documentation une fois :

```
cd docs
make dirhtml
```

Pour reconstruire automatiquement la documentation à chaque modification :

```
cd docs
sphinx-autobuild -b dirhtml . _build
```

## Rédaction de tutoriels

Nous utilisons Sphinx-Gallery pour construire les tutoriels dans le répertoire `docs/tutorials`. Consultez `docs/tutorials/demo.py` pour voir un exemple de tutoriel et la documentation de [Sphinx-Gallery](https://sphinx-gallery.github.io/stable/syntax.html) pour plus d'informations.

Pour convertir les notebooks Jupyter en tutoriels Python, vous pouvez utiliser [ce script](https://gist.github.com/mgoulao/f07f5f79f6cd9a721db8a34bba0a19a7).

Si vous souhaitez que Sphinx-Gallery exécute le tutoriel (ce qui ajoute des sorties et des graphiques), le nom du fichier doit commencer par `run_`. Notez que cela augmente le temps de génération, veillez donc à ce que le script ne prenne pas plus de quelques secondes à s'exécuter.