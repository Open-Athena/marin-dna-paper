"""Offline inputs; remote retrieval occurs only with FIGURE_FETCH=1."""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1] / "plot_data"

class MissingData(RuntimeError):
    pass

def local_path(source: str | Path) -> Path:
    source = str(source)
    parsed = urlparse(source)
    if parsed.scheme not in {"s3", "https", "http"}:
        path = Path(source)
    else:
        path = ROOT / "remote" / parsed.scheme / parsed.netloc / parsed.path.lstrip("/")
    if path.is_file():
        return path
    if os.getenv("FIGURE_FETCH") != "1" or not parsed.scheme:
        raise MissingData(f"Missing input: {path}\nOriginal source: {source}\nUse make fetch-figure-<name> with appropriate credentials, or place the snapshot at this path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        if parsed.scheme == "s3":
            import boto3
            boto3.client("s3", region_name="us-east-2").download_file(parsed.netloc, parsed.path.lstrip("/"), str(temporary))
        else:
            with urlopen(source) as response:
                temporary.write_bytes(response.read())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

def read_polars(source, **kwargs):
    kwargs.pop("storage_options", None)
    return pl.read_parquet(local_path(source), **kwargs)

def read_pandas(source, **kwargs):
    kwargs.pop("storage_options", None)
    return pd.read_parquet(local_path(source), **kwargs)

def _encode(value):
    if isinstance(value, pd.DataFrame):
        return {"type": "frame", "data": value.to_json(orient="split", double_precision=15)}
    if isinstance(value, pd.Series):
        return {"type": "series", "data": value.to_json(orient="split", double_precision=15)}
    if isinstance(value, np.ndarray):
        return {"type": "array", "data": value.tolist()}
    if isinstance(value, tuple):
        return {"type": "tuple", "data": [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value

def _decode(value):
    from io import StringIO
    if not isinstance(value, dict): return value
    kind = value.get("type")
    if kind in {"series", "frame"}:
        return pd.read_json(StringIO(value["data"]), orient="split", typ="series" if kind == "series" else "frame")
    if kind == "array": return np.asarray(value["data"])
    if kind == "tuple": return tuple(_decode(v) for v in value["data"])
    return {k: _decode(v) for k, v in value.items()}

def snapshot(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            path = ROOT / "wandb" / f"{name}.json"
            if path.exists(): return _decode(json.loads(path.read_text()))
            if os.getenv("FIGURE_FETCH") != "1":
                raise MissingData(f"Missing W&B snapshot: {path}. Run make fetch-figure-{name} with W&B credentials; query definitions are in {function.__module__}.")
            value = function(*args, **kwargs)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_encode(value)))
            return value
        return wrapped
    return decorate
