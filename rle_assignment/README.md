
# RLE Mini-Challenge

Ziel dieser Mini-Challenge ist es einen Deep Reinforcemen Learning Agenten zu trainieren, der einen möglichst hohen Score im Atari Spiel "Space Invaders" erreicht.

In diesem Repository findet ihr ein Template, auf dem ihr eure Lösung implementieren könnt, sowie eine Beispiel-Implementation eines einfachen DQN Agenten.

## Atari Space Invaders Environment

![](https://www.gymlibrary.ml/_images/space_invaders.gif)

Spiel Beschreibung: [https://atariage.com/manual_html_page.php?SoftwareLabelID=460](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)

Gym GitHub: [https://github.com/openai/gym](https://github.com/openai/gym)

Gym Dokumentation: [https://www.gymlibrary.ml](https://www.gymlibrary.ml)

Gym Space Invaders Dokumentation: [https://www.gymlibrary.ml/environments/atari/space_invaders/](https://www.gymlibrary.ml/environments/atari/space_invaders/)


## Installation

```
pip install gym[atari,accept-rom-license]==0.21.0 ale-py==0.7.4 pygame tensorboard opencv-python absl-py tensorboardX
```

Für `dqn_example.py` muss ausserdem PyTorch installiert werden:

[https://pytorch.org/get-started](https://pytorch.org/get-started)


## Inhalt

### run.py

Template für das Implementieren der Lösung.

Es steht euch jedoch offen, ob ihr dieses Template verwendet oder einen eigenen Ansatz verfolgt.

### dqn_example.py

Beispiel-Implementation eines einfachen DQN agent.

### rle_assignment/env.py

Beinhaltet die `make_env` Funktion, zum erstellen einer Environment-Instanz.

### rle_assignment/utils.py

Beinhaltet nützliche Funktionen und Klassen.
