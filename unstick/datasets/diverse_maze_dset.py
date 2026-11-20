from pathlib import Path
from .point_maze_dset import PointMazeDataset
from .traj_dset import get_train_val_sliced


def load_diverse_maze_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/diverse_maze",
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_preprocessed=False,
    use_frame_files=False,
):
    """Diverse PointMaze data from data_path/train and data_path/val (different layouts); else a random split."""
    data_path = Path(data_path)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not (train_dir.exists() and val_dir.exists()):  # single folder (finetune.py --norm-stats medium)
        dset = PointMazeDataset(
            data_path=str(data_path),
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset,
            train_fraction=split_ratio,
            num_frames=num_hist + num_pred,
            frameskip=frameskip,
        )
        return {'train': train_slices, 'valid': val_slices}, {'train': dset_train, 'valid': dset_val}

    dset_train = PointMazeDataset(
        data_path=str(train_dir),
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        use_preprocessed=use_preprocessed,
        use_frame_files=use_frame_files,
    )
    dset_val = PointMazeDataset(
        data_path=str(val_dir),
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
        use_preprocessed=use_preprocessed,
        use_frame_files=use_frame_files,
    )
    # Propagate normalization stats from train to val
    dset_val.action_mean = dset_train.action_mean
    dset_val.action_std = dset_train.action_std
    dset_val.state_mean = dset_train.state_mean
    dset_val.state_std = dset_train.state_std
    dset_val.proprio_mean = dset_train.proprio_mean
    dset_val.proprio_std = dset_train.proprio_std

    # Re-normalize val actions/proprios with train stats
    if normalize_action:
        # Reload raw and re-normalize with train stats
        raw_val = PointMazeDataset(
            data_path=str(val_dir),
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=False,
            use_preprocessed=use_preprocessed,
            use_frame_files=use_frame_files,
        )
        dset_val.actions = (raw_val.actions - dset_train.action_mean) / dset_train.action_std
        dset_val.proprios = (raw_val.proprios - dset_train.proprio_mean) / dset_train.proprio_std

    num_frames = num_hist + num_pred

    from .traj_dset import TrajSlicerDataset
    train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
    val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

    datasets = {'train': train_slices, 'valid': val_slices}
    traj_dset = {'train': dset_train, 'valid': dset_val}
    return datasets, traj_dset
