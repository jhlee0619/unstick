"""(start, goal) pairs at a controlled BFS block distance on the Diverse PointMaze layouts (as in HWM/PLDM)."""
from collections import deque
from pathlib import Path

import numpy as np
import torch

OPEN_CHARS = {"O", " ", "0"}


def _parse_open_grid(maze_spec):
    lines = maze_spec.strip().split("\\")
    rows = len(lines)
    cols = len(lines[0])
    open_mask = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if lines[r][c] in OPEN_CHARS:
                open_mask[r, c] = True
    return rows, cols, open_mask


def _bfs_distances(open_mask, start):
    """{(r, c): 4-connected block distance} over the cells reachable from start."""
    rows, cols = open_mask.shape
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and open_mask[nr, nc]:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    queue.append((nr, nc))
    return dist


def _cell_to_state(cell, rng, pos_noise=0.1):
    """(row, col) cell -> env state [x, y, 0, 0] with position jitter."""
    r, c = cell
    x = r + rng.uniform(-pos_noise, pos_noise)
    y = c + rng.uniform(-pos_noise, pos_noise)
    return np.array([x, y, 0.0, 0.0], dtype=np.float32)


def generate_diverse_maze_goals(
    val_data_path,
    n_evals,
    seed,
    min_block_radius=4,
    max_block_radius=10_000,
    pos_noise=0.1,
):
    maze_specs = torch.load(Path(val_data_path) / "maze_specs.pth")
    # maze_specs is a dict {map_idx: spec_string}; iterate in a stable order.
    map_keys = sorted(maze_specs.keys())
    num_maps = len(map_keys)
    if num_maps == 0:
        raise ValueError(f"No maze_specs found at {val_data_path}")

    rng = np.random.default_rng(seed)

    map_specs_out = []
    start_states = []
    goal_states = []
    block_dists = []
    map_indices = []

    for i in range(n_evals):
        # Round-robin over the val maps so each layout gets equal representation.
        map_idx = map_keys[i % num_maps]
        spec = maze_specs[map_idx]
        _, _, open_mask = _parse_open_grid(spec)
        open_cells = [tuple(x) for x in zip(*np.where(open_mask))]

        # start cell with a far-enough target (64 tries; else the farthest target from the last start)
        chosen = None
        for _ in range(64):
            start = open_cells[rng.integers(len(open_cells))]
            dist = _bfs_distances(open_mask, start)
            in_band = [
                cell
                for cell, d in dist.items()
                if cell != start and min_block_radius <= d <= max_block_radius
            ]
            if in_band:
                target = in_band[rng.integers(len(in_band))]
                chosen = (start, target, dist[target])
                break
        if chosen is None:
            # Relax: take the farthest reachable cell from the last start.
            dist = _bfs_distances(open_mask, start)
            reachable = [(cell, d) for cell, d in dist.items() if cell != start]
            if not reachable:
                # Degenerate map (isolated start); reuse start as goal.
                target, d = start, 0
            else:
                max_d = max(d for _, d in reachable)
                far = [cell for cell, d in reachable if d == max_d]
                target = far[rng.integers(len(far))]
                d = max_d
            chosen = (start, target, d)

        start, target, bdist = chosen
        map_specs_out.append(spec)
        start_states.append(_cell_to_state(start, rng, pos_noise))
        goal_states.append(_cell_to_state(target, rng, pos_noise))
        block_dists.append(bdist)
        map_indices.append(int(map_idx))

    return {
        "map_specs": map_specs_out,
        "start_states": np.stack(start_states),
        "goal_states": np.stack(goal_states),
        "block_dists": np.array(block_dists, dtype=np.int64),
        "map_indices": np.array(map_indices, dtype=np.int64),
    }
