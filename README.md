Learning to Drive Small Scale Cars from Scratch -- with Dreamer
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This codebase contains the code to learn to drive a Donkey Car from images using model-based reinforcement learning. This approach follows the Dreamer algorithm, which learns a world model to predict future latent states, and learns policy and value function purely based on latent imagination. This implementation is able to learn to follow a track in 5 minutes of driving around on a track which corresponds to about 6000 samples from the environment.


Core files
`models.py` contains all models used in our experiments, including the world model, actor model and value model. `agent.py` includes the dreamer agent. `dreamer.py` contains the code for using the agent to drive in the environment as well as training the agent.

Running the code
Install the required libraries:

- Python 3
- [Gym](https://gym.openai.com/)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch >= v1.5.0](http://pytorch.org/)
- [Donkeycar package](https://github.com/ari-viitala/RLDonkeyCar)


For running the experiments, please refers https://github.com/ari-viitala/donkeycar/tree/master.

References:
------------

[1] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)  

[2] [Tensorflow implementation, with tensorflow1.0](https://github.com/google-research/dreamer)

[3] [Tensorflow implementation, with tensorflow2.0](https://github.com/danijar/dreamer)

[4] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  

[5] PlaNet implementation from [@Kaixhin](https://github.com/Kaixhin) 

[6] [Donkeycar](https://www.donkeycar.com/) 

