
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFilter, ImageEnhance
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# --- Configuration ---
CLUB = "St. Pauli"  # Default club
DPI = 150                  # Resolution (100-150 is good for 1080p, 300 is overkill)
FPS = 30                   # Standard video FPS
FRAMES_PER_MINUTE = 10     # How many frames to animate one match-minute growth
OUTPUT_FILENAME = f"{CLUB}_shots_animation_hw.mp4"

# --- Setup Paths ---
# Handle team name variations for image loading
team_for_foto = CLUB
if CLUB == "RasenBallsport Leipzig": team_for_foto = "RB Leipzig"
elif CLUB == "FC Cologne": team_for_foto = "FC Köln"
elif CLUB == "Augsburg": team_for_foto = "FC Augsburg"
elif CLUB == "Borussia M.Gladbach": team_for_foto = "Borussia Mönchengladbach"

foto_path = f'images/{team_for_foto}.png'
data_path = f'{CLUB}_seasons_shots.csv'

# --- 1. Efficient Image Processing ---
def prepare_background_image(image_path, target_width, target_height):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_width, target_height), Image.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(5)) # Blur
        img = ImageEnhance.Brightness(img).enhance(1.2) # Brighten
        img = ImageEnhance.Color(img).enhance(0.6)      # Desaturate
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not load background image: {e}")
        return None

# --- 2. Hardware Encoder Detection ---
def get_ffmpeg_writer():
    # Detect hardware acceleration
    try:
        res = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
        encoders = res.stdout
        
        if 'h264_nvenc' in encoders:
            print("Encoder: NVIDIA NVENC (Hardware)")
            codec, extra = 'h264_nvenc', ['-preset', 'fast']
        elif 'h264_qsv' in encoders:
            print("Encoder: Intel QSV (Hardware)")
            codec, extra = 'h264_qsv', ['-preset', 'fast']
        elif 'h264_amf' in encoders:
            print("Encoder: AMD AMF (Hardware)")
            codec, extra = 'h264_amf', ['-usage', 'transcoding', '-quality', 'speed']
        else:
            print("Encoder: CPU (libx264) - Slower")
            codec, extra = 'libx264', ['-preset', 'fast']
            
    except FileNotFoundError:
        print("FFmpeg not found. Using low-quality Pillow writer.")
        return None

    return animation.FFMpegWriter(
        fps=FPS,
        bitrate=5000,
        codec=codec,
        extra_args=extra + ['-pix_fmt', 'yuv420p']
    )

def main():
    print(f"Processing animation for: {CLUB}")
    
    # Load Data
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Pre-calculate data series
    max_minute = int(df['minute'].max())
    shot_counts_series = df.groupby('minute').size()
    
    # Ensure all minutes 0..max exist
    minutes_x = np.arange(max_minute + 1)
    full_counts_y = np.array([shot_counts_series.get(m, 0) for m in minutes_x])
    
    # Cumulative target values for "grown" state
    # We want to animate the *counts at that minute* growing, not cumulative sum
    # The previous code animated the stem height for the current minute from 0 -> value
    
    # Setup Figure (Single Global Instance)
    fig_width, fig_height = 16, 9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI, facecolor='#f8f8fa')
    fig.tight_layout(pad=2.08)
    
    # Static Elements (Draw Once)
    # Background Image
    pixel_w, pixel_h = int(fig_width*DPI), int(fig_height*DPI)
    bg_arr = prepare_background_image(foto_path, pixel_w, pixel_h)
    
    max_y = full_counts_y.max()
    
    if bg_arr is not None:
        ax.imshow(bg_arr, aspect='auto', extent=[0, max_minute+1, 0, max_y+1], alpha=0.15, zorder=0)
    
    ax.add_patch(plt.Rectangle((0, 0), max_minute+1, max_y+1, color='white', alpha=0.3, zorder=1))
    ax.grid(True, linestyle='--', alpha=0.6, color='gray', zorder=1)
    
    ax.set_xlim(0, max_minute + 1)
    ax.set_ylim(0, max_y + 1)
    ax.set_xlabel('Minutes - all matches', weight='bold', size=12)
    ax.set_ylabel('Number of Shots', weight='bold', size=12)
    ax.set_title(f'{team_for_foto} Shot Count per Minute', weight='bold', size=15)

    # Dynamic Elements (To Be Updated)
    # Initialize stem plot with zeros
    markerline, stemlines, baseline = ax.stem(minutes_x, np.zeros_like(full_counts_y), 
                                             linefmt='m-', markerfmt='o', basefmt='k-')
    
    plt.setp(markerline, markersize=8, color='orange', zorder=3)
    plt.setp(stemlines, linewidth=2, color='magenta', zorder=2)
    plt.setp(baseline, visible=False)

    # Text elements
    title_text = ax.text(max_minute * 0.5, max_y * 1.05, "", ha='center', weight='bold', size=12, transform=ax.transData)
    total_text = ax.text(max_minute * 0.7, max_y * 0.9, "", weight='bold', size=15, 
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'), zorder=4)
    
    # --- Animation Logic (Blitting Optimized) ---
    def init():
        # Return artists that change
        return markerline, stemlines, total_text

    def update(frame_idx):
        # Determine minute and sub-minute progress
        minute_idx = int(frame_idx)
        if minute_idx >= len(minutes_x): minute_idx = len(minutes_x) - 1
        
        progress = frame_idx - int(frame_idx)
        
        # Construct current Y values
        # All past minutes have full value
        current_y = full_counts_y.copy()
        
        # Future minutes are 0
        current_y[minute_idx+1:] = 0
        
        # Current minute is growing
        if minute_idx < len(minutes_x):
            current_y[minute_idx] = full_counts_y[minute_idx] * progress

        # Update Stem Lines efficiently
        # stemlines is a LineCollection. We update segments: [[x, 0], [x, y]]
        new_segments = []
        for x, y in zip(minutes_x, current_y):
            new_segments.append(np.array([[x, 0], [x, y]]))
        
        stemlines.set_segments(new_segments)
        
        # Update Markers
        markerline.set_data(minutes_x, current_y)
        
        # Update Text
        total_text.set_text(f"Total Shots: {int(current_y.sum())}")
        
        return markerline, stemlines, total_text

    # Calculate Frames
    # frames = minutes * frames_per_minute
    total_frames = int((max_minute + 1) * FRAMES_PER_MINUTE)
    # Add freeze frames at end
    freeze_frames = int(7 * FPS) # 3 seconds freeze
    
    # Frame generator
    frames = np.linspace(0, max_minute, total_frames)
    # Append freeze frames (repeating the last value)
    frames = np.concatenate([frames, [max_minute] * freeze_frames])
    
    print(f"Generating {len(frames)} frames...")
    
    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, 
                                  blit=True, interval=1000/FPS)
    
    # Save
    writer = get_ffmpeg_writer()
    if writer:
        total_frames_count = len(frames)
        print(f"Starting render of {total_frames_count} frames...")
        
        if tqdm:
            with tqdm(total=total_frames_count, unit="frames") as pbar:
                # Callback is called with (current_frame_number, total_frames)
                # We update by 1 for each call
                def progress_callback(frame, total):
                    pbar.update(1)
                ani.save(OUTPUT_FILENAME, writer=writer, dpi=DPI, progress_callback=progress_callback)
        else:
            # Simple fallback
            def progress_callback(frame, total):
                if frame % 50 == 0 or frame == total - 1:
                    print(f"Rendering: {frame}/{total} frames ({(frame/total)*100:.1f}%)", end='\r')
            ani.save(OUTPUT_FILENAME, writer=writer, dpi=DPI, progress_callback=progress_callback)
            print("\nDone.")

        print(f"Success! Saved to {OUTPUT_FILENAME}")
    else:
        print("Encoding failed.")

if __name__ == "__main__":
    main()
