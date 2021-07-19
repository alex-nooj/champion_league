# MIT License. 2020. Joe Tatusko.
from os import path
from typing import List
from typing import Optional

from omegaconf import DictConfig
from omegaconf import OmegaConf


def load_cfg(base_cfg_path: str, script_cfg_path: str) -> DictConfig:
    # Load base config
    base_conf = OmegaConf.merge(_conf_from_path(base_cfg_path), _conf_from_path(script_cfg_path))
    cli_conf = OmegaConf.from_cli()
    resume_conf = _get_special_conf(cli_conf, "resume")
    user_conf = _get_special_conf(cli_conf, "config")

    # Remove special fields
    cli_conf = _rm_fields(cli_conf, ["resume", "config"])

    # Merge first pass
    return _merge_first(base_conf, cli_conf, resume_conf, user_conf)


def _conf_from_path(script_cfg_path: str) -> DictConfig:
    if path.exists(script_cfg_path):
        base_conf = OmegaConf.load(script_cfg_path)
    else:
        base_conf = OmegaConf.create()
    return base_conf


def _get_special_conf(conf: DictConfig, field_name: str) -> Optional[DictConfig]:
    if field_name in conf:
        special_conf = OmegaConf.load(path.abspath(conf[field_name]))
    else:
        special_conf = None
    return special_conf


def _merge_first(
    base_conf: DictConfig,
    cli_conf: DictConfig,
    resume_conf: Optional[DictConfig],
    user_conf: Optional[DictConfig],
) -> DictConfig:
    if resume_conf:
        conf = OmegaConf.merge(resume_conf, cli_conf)
    elif user_conf:
        conf = OmegaConf.merge(base_conf, user_conf, cli_conf)
    else:
        conf = OmegaConf.merge(base_conf, cli_conf)
    return conf


def _merge_second(
    merge_conf: DictConfig,
    cli_conf: DictConfig,
    user_conf: Optional[DictConfig],
    module_base_confs: List[DictConfig],
) -> DictConfig:
    user_confs = [user_conf] if user_conf else []
    return OmegaConf.merge(merge_conf, *module_base_confs, *user_confs, cli_conf)


def _rm_fields(conf: DictConfig, field_names: List[str]) -> DictConfig:
    return OmegaConf.create({k: v for k, v in conf.items() if k not in field_names})
