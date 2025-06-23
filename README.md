# PPO for Gymnasium

This repository contains a PyTorch implementation of the Proximal Policy Optimization (PPO) algorithm for training agents in Gymnasium environments.

## Features

-   Configurable hyperparameters via a YAML file.
-   Logging to TensorBoard and Weights & Biases.
-   GPU-accelerated training.
-   Episodic video recording.
-   A utility to watch trained agents.

## File Structure

```
.
├── README.md
├── config.yaml
├── main.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── agent.py
    ├── train.py
    └── utils.py
```

## Usage

1.  **Installation**

    It is recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration**

    The `config.yaml` file contains all the hyperparameters for training. You can modify this file to experiment with different settings.

3.  **Training**

    To start training, run the following command:

    ```bash
    python main.py
    ```

    If you want to use Weights & Biases for logging, you need to log in first:

    ```bash
    wandb login
    ```

    Then, set the `wandb_project_name` and `wandb_entity` in your `config.yaml`.

4.  **Watching a Trained Agent**

    The training script saves videos of the agent's performance in the `videos` directory. To watch a saved episode, run the `watch.py` script and provide the path to the video file:

    ```bash
    python watch.py videos/<run_name>/rl-video-episode-0.mp4
    ```

    Replace `<run_name>` with the actual run name of your trained agent. The run name is printed to the console when you start training.
