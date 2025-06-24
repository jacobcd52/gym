#!/bin/bash

# Setup script for Gymnasium environments
# This script installs system dependencies and all Gymnasium environment packages

set -e  # Exit on any error

echo "🚀 Setting up Gymnasium environments..."

# Update package list
echo "📦 Updating package list..."
apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    swig \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install Python dependencies
echo "🐍 Installing Python dependencies..."

# Install basic Gymnasium with all extras
echo "📚 Installing Gymnasium with all environments..."
pip install gymnasium[all]

# Install additional environment packages that might not be included in [all]
echo "🎮 Installing additional environment packages..."
pip install \
    gymnasium[atari] \
    gymnasium[box2d] \
    gymnasium[mujoco] \
    gymnasium[classic_control] \
    gymnasium[mujoco_py] \
    gymnasium[robotics] \
    gymnasium[toy_text] \
    gymnasium[accept-rom-license]

# Install additional useful packages
echo "🔧 Installing additional useful packages..."
pip install \
    supersuit \
    pettingzoo \
    shimmy

echo "✅ Setup complete! You should now have access to many more Gymnasium environments."
echo ""
echo "To verify, run:"
echo "python -c \"import gymnasium as gym; print('Total environments:', len(list(gym.envs.registry.keys())))\""
echo ""
echo "Available environment categories:"
echo "- Classic Control: CartPole, Pendulum, Acrobot, MountainCar"
echo "- Box2D: BipedalWalker, CarRacing, LunarLander"
echo "- MuJoCo: Humanoid, Ant, HalfCheetah, Hopper, Walker2d, Swimmer"
echo "- Atari: Pong, Breakout, SpaceInvaders, and many more"
echo "- Toy Text: Blackjack, Taxi, FrozenLake, CliffWalking"
echo "- Robotics: Fetch, Hand, ShadowHand environments" 