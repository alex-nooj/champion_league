import os


def check_and_make_dir(path: str) -> None:
    """First checks if a directory exists then creates it if it doesn't.

    Parameters
    ----------
    path: str
        The path of the directory to be created

    Returns
    -------
    None
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def get_save_dir(logdir: str, tag: str, epoch: int) -> str:
    return os.path.join(logdir, tag, f"{tag}_{epoch:05d}")


class DotDict(dict):
    """
    Dictionary to access attributes
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # Support pickling
    def __getstate__(obj):
        return dict(obj.items())

    def __setstate__(cls, attributes):
        return DotDict(**attributes)


def get_most_recent_epoch(agent_path: str) -> int:
    epochs = [
        int(e.rsplit("_")[-1])
        for e in os.listdir(agent_path)
        if os.path.isdir(os.path.join(agent_path, e)) and e != "sl"
    ]

    return max(epochs)
