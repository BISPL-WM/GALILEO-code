"""Dataclass configs from YAML files and `key.sub=value` overrides."""
import dataclasses
import typing

import yaml


def _convert(tp, value):
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(value, dict):
        return from_dict(tp, value)
    if origin in (tuple, typing.Tuple) and isinstance(value, (list, tuple)):
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        return tuple(value)
    if origin is typing.Literal:
        return value
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return value if value is None or len(args) != 1 else _convert(args[0], value)
    if tp in (int, float, str, bool) and value is not None and not isinstance(value, tp):
        if tp is bool and isinstance(value, str):
            return value.lower() in ("1", "true", "yes")
        return tp(value)
    return value


def from_dict(cls, d):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**{k: _convert(hints[k], v) for k, v in d.items()})


def set_path(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load(cls, path=None, overrides=()):
    """cls(**yaml) with `a.b=v` overrides (values parsed as YAML scalars)."""
    d = yaml.safe_load(open(path)) if path else {}
    d = d or {}
    for kv in overrides:
        k, v = kv.split("=", 1)
        set_path(d, k, yaml.safe_load(v))
    return from_dict(cls, d)


def to_dict(cfg):
    return dataclasses.asdict(cfg)
