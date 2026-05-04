#!/usr/bin/env python3
"""
animate_traffic.py

Reads traffic_anim_data.csv produced by:
    mpirun -np N ./traffic_circle_mpi <iter> 4 --anim

Generates traffic_animation.gif showing the 16-slot roundabout
(4 roads, textbook Figure 10.21 layout) frame by frame.

Usage:
    python3 animate_traffic.py [--input FILE] [--output FILE] [--fps N]
"""

import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── geometry ────────────────────────────────────────────────────────────────
NUM_SLOTS  = 16
ROAD_SLOTS = [0, 4, 8, 12]          # entrance/exit slot indices: N, W, S, E
ROAD_NAMES = ['N', 'W', 'S', 'E']

# Color per destination slot (-1 = empty)
DEST_COLOR = {
    -1: '#cccccc',   # empty
     0: '#e74c3c',   # heading to N  (red)
     4: '#3498db',   # heading to W  (blue)
     8: '#2ecc71',   # heading to S  (green)
    12: '#f39c12',   # heading to E  (orange)
}
ROAD_COLORS = [DEST_COLOR[s] for s in ROAD_SLOTS]

RING_R    = 1.0    # radius of the slot ring
SLOT_R    = 0.13   # visual radius of each slot circle


def slot_angle(i):
    """Angle (radians) of slot i: slot 0 at top, clockwise."""
    return np.pi / 2 - 2 * np.pi * i / NUM_SLOTS


def slot_xy(i, r=RING_R):
    a = slot_angle(i)
    return r * np.cos(a), r * np.sin(a)


# ── data loading ─────────────────────────────────────────────────────────────
def load_csv(path):
    frames = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append({
                'iter':   int(row['iter']),
                'circle': [int(row[f's{i}']) for i in range(NUM_SLOTS)],
                'queue':  [int(row[f'q{i}']) for i in range(4)],
            })
    return frames


# ── figure setup ─────────────────────────────────────────────────────────────
def build_figure():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    fig.patch.set_facecolor('#f8f8f8')
    ax.set_facecolor('#f8f8f8')

    # outer ring guide
    ring = plt.Circle((0, 0), RING_R, fill=False, color='#aaaaaa',
                       linewidth=1.5, linestyle='--')
    ax.add_patch(ring)

    # direction arrow showing clockwise flow
    theta = np.linspace(np.pi * 0.55, np.pi * 0.45, 60)
    ax.annotate('', xy=(RING_R * np.cos(theta[-1]), RING_R * np.sin(theta[-1])),
                xytext=(RING_R * np.cos(theta[0]),  RING_R * np.sin(theta[0])),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))

    # road labels
    for i, (name, slot) in enumerate(zip(ROAD_NAMES, ROAD_SLOTS)):
        x, y = slot_xy(slot, r=RING_R * 1.42)
        ax.text(x, y, name, ha='center', va='center',
                fontsize=15, fontweight='bold', color=ROAD_COLORS[i])

    # slot circles (drawn once, coloured in update())
    slot_patches = []
    for i in range(NUM_SLOTS):
        x, y = slot_xy(i)
        p = plt.Circle((x, y), SLOT_R, color=DEST_COLOR[-1], zorder=3)
        ax.add_patch(p)
        # small slot-index label
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=6, color='white', fontweight='bold', zorder=4)
        slot_patches.append(p)

    # queue bars: one Line2D per entrance, pointing radially outward
    queue_artists = []
    for i, slot in enumerate(ROAD_SLOTS):
        x, y = slot_xy(slot)
        length = np.hypot(x, y)
        nx, ny = x / length, y / length    # outward unit vector
        line, = ax.plot([], [], color=ROAD_COLORS[i],
                        linewidth=10, solid_capstyle='round',
                        alpha=0.7, zorder=2)
        queue_artists.append((line, x, y, nx, ny))

    # queue length numeric labels
    queue_texts = []
    for i, slot in enumerate(ROAD_SLOTS):
        x, y = slot_xy(slot, r=RING_R * 1.75)
        t = ax.text(x, y, '', ha='center', va='center',
                    fontsize=9, color=ROAD_COLORS[i], fontweight='bold')
        queue_texts.append(t)

    # title
    title = ax.set_title('Iteration: 0', fontsize=11, pad=8)

    # legend
    handles = [mpatches.Patch(color=DEST_COLOR[s], label=f'→ {n}')
               for s, n in zip(ROAD_SLOTS, ROAD_NAMES)]
    handles.append(mpatches.Patch(color=DEST_COLOR[-1], label='empty'))
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.7, borderpad=0.6)

    return fig, ax, slot_patches, queue_artists, queue_texts, title


# ── animation ────────────────────────────────────────────────────────────────
def make_animation(frames, output_path, fps):
    fig, ax, slot_patches, queue_artists, queue_texts, title = build_figure()

    def update(frame_idx):
        row    = frames[frame_idx]
        circle = row['circle']
        queues = row['queue']

        # update slot colours
        for i, patch in enumerate(slot_patches):
            patch.set_facecolor(DEST_COLOR.get(circle[i], DEST_COLOR[-1]))

        # update queue bars and labels
        for i, (line, x, y, nx, ny) in enumerate(queue_artists):
            q = queues[i]
            if q > 0:
                bar_len = min(q * 0.07, 0.55)   # cap so it doesn't overflow
                x0 = x + nx * (SLOT_R + 0.03)
                x1 = x0 + nx * bar_len
                y0 = y + ny * (SLOT_R + 0.03)
                y1 = y0 + ny * bar_len
                line.set_data([x0, x1], [y0, y1])
                queue_texts[i].set_text(str(q))
            else:
                line.set_data([], [])
                queue_texts[i].set_text('')

        title.set_text(f'Iteration: {row["iter"]}')
        artists = slot_patches + [la[0] for la in queue_artists] + queue_texts + [title]
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 / fps, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f'Saved {len(frames)}-frame animation → {output_path}')


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Animate traffic circle simulation data')
    ap.add_argument('--input',  default='traffic_anim_data.csv',
                    help='CSV file produced by --anim mode (default: traffic_anim_data.csv)')
    ap.add_argument('--output', default='traffic_animation.gif',
                    help='Output GIF path (default: traffic_animation.gif)')
    ap.add_argument('--fps',    type=int, default=15,
                    help='Frames per second (default: 15)')
    args = ap.parse_args()

    frames = load_csv(args.input)
    print(f'Loaded {len(frames)} frames from {args.input}')
    make_animation(frames, args.output, args.fps)
