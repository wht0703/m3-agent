import argparse
import subprocess
import os
from tqdm import tqdm

TMP_dir = "tmp/"
os.makedirs(TMP_dir, exist_ok=True)

def print_clip(mem_path, clip_id, orginal=False):
    result = subprocess.run(
        ['python', 'visualization.py', '--mem_path', f'{mem_path}', '--clip_id', f'{clip_id}'],
        capture_output=True, text=True
    )
    if not orginal:
        save_path = f'{TMP_dir}/clip_test_{clip_id}.txt'
    else:
        save_path = f'{TMP_dir}/clip_original_{clip_id}.txt'   
    with open(save_path, 'w') as f:
        f.write(result.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="data/memory_graphs/robot/bedroom_01.pkl")
    parser.add_argument("--video_length", type=int, default=0)
    parser.add_argument("--original", type=bool, default=False)
    args = parser.parse_args()

    with tqdm(range(args.video_length), desc="Processing clips") as pbar:
        for clip_id in pbar:
            pbar.set_description(f"Processing clip id {clip_id}")
            print_clip(args.mem_path, clip_id, args.original)
        