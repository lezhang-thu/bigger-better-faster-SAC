# Generalizing soft actor-critic algorithms to discrete action spaces 

This repository implements the SAC-BBF agent in JAX (to be presented in **PRCV 2024**), building
on [BBF](https://github.com/google-research/google-research/tree/master/bigger_better_faster).

## Setup
To install the repository, simply run `pip install -r requirements.txt`. **Tested only on Python 3.10**.


## Training
To run a BBF agent locally, run

```
bash run-cuda0.sh
```

## References
* [Schwarzer, Max, et al. "Bigger, better, faster: Human-level atari with human-level efficiency." International Conference on Machine Learning. PMLR, 2023.][bbf]
* [Max Schwarzer, Ankesh Anand, Rishab Goel, Devon Hjelm, Aaron Courville and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In The Ninth International Conference on Learning Representations, 2021.][spr]

* [Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc Bellemare, Aaron Courville.  Sample-efficient reinforcement learning by breaking the replay ratio barrier. In The Eleventh International Conference on Learning Representations, 2023][sr-spr]

[bbf]: https://proceedings.mlr.press/v202/schwarzer23a.html
[spr]: https://openreview.net/forum?id=uCQfPZwRaUu
[sr-spr]: https://openreview.net/forum?id=OpC-9aBBVJe
