import pickle
from pathlib import Path
from typing import Any, Dict, Union

from net import Net


def save_bp_checkpoint(
    path: Union[str, Path],
    net: Net,
    scaler_X,
    random_state: int,
    task: str,
    scaler_y=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": 1,
        "task": task,
        "random_state": random_state,
        "net_state": net.get_state(),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_bp_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
