import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_trajectory(trajectory_path):
    """
    Visualizes a saved trajectory with an interactive slider.
    """
    try:
        data = np.load(trajectory_path)
        observations = data['observations']
        advantages = data['advantages']
    except FileNotFoundError:
        st.error(f"Trajectory file not found at: {trajectory_path}")
        return

    st.title("Trajectory Visualization")

    # Frame slider
    num_frames = len(observations)
    if num_frames == 0:
        st.warning("Trajectory is empty.")
        return
        
    frame_idx = st.slider("Frame", 0, num_frames - 1, 0)

    # Display observation
    st.subheader(f"Frame {frame_idx}")
    
    # Handle different observation shapes
    obs = observations[frame_idx]
    if len(obs.shape) == 3 and (obs.shape[0] == 4 or obs.shape[2] == 4): # Stacked frames for Atari
        # Assuming channels-first (C, H, W) for FrameStack
        if obs.shape[0] == 4:
            # Display frames in a row
            st.image([obs[i] for i in range(4)], width=150, caption=[f"Frame {i}" for i in range(4)])
        else: # Assuming channels-last (H, W, C)
            st.image(obs, caption="Observation", use_column_width=True)

    elif len(obs.shape) == 2 or (len(obs.shape) == 3 and obs.shape[2] in [1, 3]): # Grayscale or RGB
        st.image(obs, caption="Observation", use_column_width=True)
    else: # Vector-based observation
        st.text("Vector Observation:")
        st.write(obs)


    # Plot advantages
    st.subheader("Advantages vs. Timestep")
    fig, ax = plt.subplots()
    ax.plot(advantages)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Advantage (GAE)")
    ax.axvline(x=frame_idx, color='r', linestyle='--', label=f'Current Timestep: {frame_idx}')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PPO agent's trajectory.")
    parser.add_argument("trajectory_path", type=str, help="Path to the .npz trajectory file.")
    
    # Check if running in Streamlit's context. If so, don't parse args from command line.
    try:
        # This is a bit of a hack to get the script argument in streamlit
        args = parser.parse_args()
        trajectory_path = args.trajectory_path
    except SystemExit:
        # This will be raised by argparse in Streamlit environment.
        # We assume the path is provided through some other means if not command line
        # For development, you might hardcode a path or use a file uploader.
        
        # Simple solution: Use a file_uploader if no arg is passed
        st.sidebar.title("Upload Trajectory")
        uploaded_file = st.sidebar.file_uploader("Choose a .npz file", type="npz")
        
        if uploaded_file is not None:
            # To use the uploader, we need to save the file temporarily
            with open("temp_trajectory.npz", "wb") as f:
                f.write(uploaded_file.getbuffer())
            trajectory_path = "temp_trajectory.npz"
        else:
            st.info("Please upload a trajectory file to begin.")
            st.stop()
            
    visualize_trajectory(trajectory_path)
    
    # Clean up temp file if it exists
    if 'trajectory_path' in locals() and trajectory_path == "temp_trajectory.npz" and os.path.exists(trajectory_path):
        os.remove(trajectory_path) 