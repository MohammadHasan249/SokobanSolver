import copy

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from snowman import snowman_goal_state
from solution import (anytime_gbfs, anytime_weighted_astar, heur_alternate,
                      heur_manhattan_distance)
from test_problems import PROBLEMS

# Windows :(. I let imagemagick install ffmpeg. if you have ffmpeg installed, use the path that
# you get from typing `where ffmpeg` into the terminal
# Imagemagick: https://imagemagick.org/script/download.php
plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\ffmpeg.exe'


man_dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
def modified_manhattan(state):
    if snowman_goal_state(state):
        return 0
    score = 0
    for c, d in state.snowballs.items():
        score += man_dist(c, state.destination) * (3 - d)
        score += man_dist(c, state.robot) * (3 - d) * 0.2
    return score


# ======================== OPTIONS ===================================

VIS_PROBLEMS = [8]
TIMEBOUND = 5
HEURISTIC_NAME = ['manhattan', 'modified manhattan', 'alternate'][1]
FORMAT = ['gif', 'mp4'][0]

# ====================================================================

HEURISTICS = {
    'manhattan': heur_manhattan_distance,
    'alternate': heur_alternate,
    'modified manhattan': modified_manhattan,
}
HEURISTIC = HEURISTICS[HEURISTIC_NAME]
textbbox = {
    'facecolor': 'white',
    'alpha': 0.5,
}


def plot_frame(i, ax, cmap, frames, positions, title=''):
    i = min(i, len(positions) - 1)
    ax.clear()
    ax.set_title(f'{title} -- {len(positions)}\n{i}')
    ax.imshow(frames[i], cmap=cmap)
    ax.text(positions[i]['robot'][0]+1, positions[i]['robot'][1]+1,
            'R', ha='center', va='center', bbox=textbbox)
    ax.text(positions[i]['target'][0]+1, positions[i]['target'][1]+1,
            'T', ha='center', va='center', bbox=textbbox)
    for (x, y), b in positions[i]['sballs'].items():
        ax.text(x+1, y+1, b, ha='center', va='center', bbox=textbbox)


for PROB in VIS_PROBLEMS:
    prb = PROBLEMS[PROB]
    astar_sol = anytime_weighted_astar(
        prb, heur_fn=HEURISTIC, weight=100, timebound=TIMEBOUND)
    gbfs_sol = anytime_gbfs(prb, heur_fn=HEURISTIC, timebound=TIMEBOUND)
    cols = prb.width*2 + 1 + 2
    rows = prb.height

    fig = plt.figure(figsize=(12, 12 * rows / cols))
    gs = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
    ax1 = fig.add_subplot(gs[:, :prb.width])
    ax2 = fig.add_subplot(gs[:, prb.width+1:2*prb.width+1])
    cax = fig.add_subplot(gs[:, 2*prb.width+2:])

    fig.suptitle(f'Problem {PROB}\n{HEURISTIC_NAME}', fontsize=14)
    for a in [ax1, ax2]:
        a.set_axis_off()

    def make_frames(state, initial):
        if not state:
            state = initial
        depth = 1
        tstate = state
        while tstate.parent:
            depth += 1
            tstate = tstate.parent
        w, h = state.width, state.height
        frames = np.zeros((depth, h, w))
        for (x, y) in state.obstacles:
            frames[:, y, x] = np.NaN
        positions = [{} for i in range(depth)]
        i = depth - 1
        while i >= 0:
            positions[i] = {
                'sballs': state.snowballs,
                'robot': state.robot,
                'target': state.destination,
            }
            initial_robot = state.robot
            f = frames[i]
            for x in range(w):
                for y in range(h):
                    if f[y, x] == 0:
                        state.robot = (x, y)
                        f[y, x] = HEURISTIC(state)
            state.robot = initial_robot
            state = state.parent
            i -= 1
        frames = np.pad(frames, ((0, 0), (1, 1), (1, 1)),
                        constant_values=np.NaN)
        return frames, positions

    astar_frames, astar_positions = make_frames(astar_sol, prb)
    gbfs_frames, gbfs_positions = make_frames(gbfs_sol, prb)

    cmap = copy.copy(mpl.cm.get_cmap('hot_r'))
    cmap.set_bad(color='black')
    mi = min(np.nanmin(astar_frames), np.nanmin(gbfs_frames))
    ma = max(np.nanmax(astar_frames), np.nanmax(gbfs_frames))
    norm = mpl.colors.Normalize(vmin=mi, vmax=ma)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label('Heuristic')

    moviewriter = animation.writers[{
        'mp4': 'ffmpeg',
        'gif': 'imagemagick',}[FORMAT]](fps=5)
    with moviewriter.saving(fig, f'{PROB}_{HEURISTIC_NAME}.{FORMAT}', dpi=64):
        for j in range(max(len(astar_frames), len(gbfs_frames))):
            plot_frame(j, ax1, cmap, gbfs_frames,
                           gbfs_positions, title='GBFS')
            plot_frame(j, ax2, cmap, astar_frames,
                           astar_positions, title='A-star')
            moviewriter.grab_frame()
        moviewriter.finish()
