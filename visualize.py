import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def visualize_trajectory(trajectory_path):
    """
    Visualizes a saved trajectory with an interactive slider and play button.
    """
    # Initialize session state
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'frame_idx' not in st.session_state:
        st.session_state.frame_idx = 0
    if 'playback_speed' not in st.session_state:
        st.session_state.playback_speed = "1x"
    if 'current_trajectory' not in st.session_state:
        st.session_state.current_trajectory = None
    if 'selected_env' not in st.session_state:
        st.session_state.selected_env = 0

    # Reset frame index if trajectory changes
    if st.session_state.current_trajectory != trajectory_path:
        st.session_state.current_trajectory = trajectory_path
        st.session_state.frame_idx = 0
        st.session_state.playing = False
        st.session_state.selected_env = 0

    try:
        data = np.load(trajectory_path)
        # States shape: (T, N, C, H, W), Advantages shape: (T, N)
        states = data['states']
        advantages = data['advantages']
        num_saved_envs = states.shape[1]
    except FileNotFoundError:
        st.error(f"Trajectory file not found at: {trajectory_path}")
        return
    except Exception as e:
        st.error(f"Failed to load or process trajectory file: {e}")
        return

    st.title("Trajectory Visualization")

    # Environment selector
    st.session_state.selected_env = st.sidebar.selectbox(
        "Select Environment to Display", 
        options=list(range(num_saved_envs)),
        index=st.session_state.get('selected_env', 0)
    )

    # Isolate the data for the selected environment
    env_states = states[:, st.session_state.selected_env]
    env_advantages = advantages[:, st.session_state.selected_env]
    
    num_frames = len(env_states)
    if num_frames == 0:
        st.warning("Trajectory is empty.")
        return

    # Layout for controls
    control_cols = st.columns([1, 1, 5])
    play_button = control_cols[0].button('▶️ Play' if not st.session_state.playing else '❚❚ Pause', use_container_width=True)
    
    speed_options = ["1x", "2x", "4x", "8x"]
    st.session_state.playback_speed = control_cols[1].selectbox("Speed", speed_options, index=speed_options.index(st.session_state.playback_speed))
    
    # Slider logic needs to be handled carefully with session state for play mode
    st.session_state.frame_idx = control_cols[2].slider("Frame", 0, num_frames - 1, st.session_state.frame_idx)

    if play_button:
        st.session_state.playing = not st.session_state.playing

    # Main layout for video and plot
    col1, col2 = st.columns(2)

    # Column 1: Display observation
    with col1:
        st.subheader(f"Frame {st.session_state.frame_idx}")
        obs = env_states[st.session_state.frame_idx] # Shape: (C, H, W)
        if len(obs.shape) == 3 and obs.shape[0] == 4: # Stacked frames for Atari
            st.image(obs[-1], caption="Current Observation", use_container_width=True)
        elif len(obs.shape) == 2 or (len(obs.shape) == 3 and obs.shape[2] in [1, 3]): # Grayscale or RGB
            st.image(obs, caption="Observation", use_container_width=True)
        else: # Vector-based observation
            st.text("Vector Observation:")
            st.write(obs)

    # Column 2: Plot advantages
    with col2:
        st.subheader("Advantages vs. Timestep")
        fig, ax = plt.subplots()
        ax.plot(env_advantages)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Advantage (GAE)")
        ax.axvline(x=st.session_state.frame_idx, color='r', linestyle='--', label=f'Current Timestep')
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    if st.session_state.playing:
        if st.session_state.frame_idx < num_frames - 1:
            st.session_state.frame_idx += 1
        else:
            st.session_state.playing = False # Stop at the end
        
        speed_multiplier = float(st.session_state.playback_speed.replace('x', ''))
        base_delay = 0.1
        time.sleep(base_delay / speed_multiplier) # Control playback speed
        st.rerun()

if __name__ == "__main__":
    st.sidebar.title("Load Trajectory")
    
    # Option to select a run
    runs_dir = "trajectories"
    if os.path.exists(runs_dir):
        # List all environment folders (like 'ALE')
        env_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        selected_env = st.sidebar.selectbox("Select an environment", sorted(env_folders))

        if selected_env:
            env_path = os.path.join(runs_dir, selected_env)
            run_folders = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
            selected_run = st.sidebar.selectbox("Select a run", sorted(run_folders, reverse=True))

            if selected_run:
                trajectory_dir = os.path.join(env_path, selected_run)
                trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith('.npz')]
                
                if trajectory_files:
                    selected_trajectory_file = st.sidebar.selectbox("Select a trajectory", sorted(trajectory_files))
                    trajectory_path = os.path.join(trajectory_dir, selected_trajectory_file)
                    st.sidebar.markdown(f"**Loaded:** `{selected_trajectory_file}`")
                    visualize_trajectory(trajectory_path)
                else:
                    st.warning(f"No trajectory files found in `{trajectory_dir}`.")
            else:
                st.info(f"No runs found in the '{selected_env}' directory.")
        else:
            st.info("No environments found in the 'trajectories' directory.")
    else:
        st.error("'trajectories' directory not found.")
        st.info("Please run training first to generate trajectories.") 