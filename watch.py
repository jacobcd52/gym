import argparse
from src.utils import watch_episode

def main():
    parser = argparse.ArgumentParser(description="Watch a saved episode.")
    parser.add_argument("video_path", type=str, help="Path to the video file to watch.")
    args = parser.parse_args()
    watch_episode(args.video_path)

if __name__ == "__main__":
    main() 