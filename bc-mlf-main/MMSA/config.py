import json
import os
from pathlib import Path
import random
import re
from easydict import EasyDict as edict


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATH_DEFAULTS = {
    "BCMLF_DATA_ROOT": _REPO_ROOT / "data" / "mmsa",
    "BCMLF_MODEL_ROOT": _REPO_ROOT / "models",
    "BCMLF_CHECKPOINT_ROOT": _REPO_ROOT / "checkpoints",
}


def _expand_config_value(value):
    """Expand ${VAR} placeholders using the environment and repo-local defaults."""
    if value is None:
        return value
    text = os.path.expanduser(str(value))

    def replace_var(match):
        name = match.group(1)
        if name in os.environ:
            return os.environ[name]
        if name in _PATH_DEFAULTS:
            return str(_PATH_DEFAULTS[name])
        raise ValueError(
            f"Environment variable '{name}' is required by the config but is not set."
        )

    text = re.sub(r"\$\{([^}]+)\}", replace_var, text)
    # Also support the native %VAR% form on Windows.
    return os.path.expandvars(text)


def _resolve_data_root(value):
    return str(Path(_expand_config_value(value)).expanduser().resolve())


def _resolve_optional_path(value, root=None):
    if not value:
        return value
    expanded = _expand_config_value(value)
    path = Path(expanded).expanduser()
    if root is not None and not path.is_absolute():
        path = Path(root) / path
    return str(path)


def get_default_regression_config_file(model_name: str, dataset_name: str) -> Path:
    """Return the canonical BC-MLF config used when --config is omitted."""
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    if model_name == "bienc":
        filename = "bienc.json"
    elif model_name == "msalm":
        filename = {
            "mosei": "large_best_bchead_eps008.json",
            "sims": "base_best_bchead_eps008.json",
        }.get(dataset_name)
        if filename is None:
            raise ValueError(f"Unsupported BC-MLF dataset: {dataset_name}")
    else:
        raise ValueError(
            f"No default BC-MLF config is defined for model '{model_name}'. "
            "Pass --config explicitly."
        )
    return Path(__file__).parent / "config" / "regression" / "bcmlf" / dataset_name / filename


def get_config_regression(
    model_name: str, dataset_name: str, config_file: str = "",
) -> dict:
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file in ("", None):
        config_file = get_default_regression_config_file(model_name, dataset_name)
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    if model_common_args['need_data_aligned'] and 'aligned' in dataset_args:
        dataset_args = dataset_args['aligned']
    else:
        dataset_args = dataset_args['unaligned']
    # transformations
    use_augmentation = model_common_args.get("use_augmentation", False)
    use_m3xup = model_common_args.get("use_m3xup", False)
    if not use_augmentation:
        model_common_args["use_augmentation"] = use_augmentation  # False
    if not use_m3xup:
        model_common_args["use_m3xup"] = use_m3xup  # False

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    data_root = _resolve_data_root(config_all['datasetCommonParams']['dataset_root_dir'])
    config['dataset_root_dir'] = data_root
    config['featurePath'] = os.path.join(data_root, config['featurePath'])
    if config_all[model_name]["datasetParams"][dataset_name].get("hfPath", False):
        config["hfPath"] = \
            os.path.join(
                data_root,
                config_all[model_name]["datasetParams"][dataset_name]["hfPath"]
            )
    if config.get("lm"):
        config["lm"] = _expand_config_value(config["lm"])
    if config.get("av_enc", {}).get("path_to_pretrained"):
        config["av_enc"]["path_to_pretrained"] = _resolve_optional_path(
            config["av_enc"]["path_to_pretrained"],
            root=_resolve_data_root("${BCMLF_CHECKPOINT_ROOT}"),
        )
    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_tune(
    model_name: str, dataset_name: str, config_file: str = "",
    random_choice: bool = True
) -> dict:
    """
    Get the tuning config of given dataset and model from config file.

    Parameters:
        model_name: Name of model.
        dataset_name: Name of dataset.
        config_file: Path to config file, if given an empty string, will use default config file.
        random_choice: If True, will randomly choose a config from the list of configs.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # random choice of args
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # nested params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    data_root = _resolve_data_root(config_all['datasetCommonParams']['dataset_root_dir'])
    config['dataset_root_dir'] = data_root
    config['featurePath'] = os.path.join(data_root, config['featurePath'])
    if config.get("lm"):
        config["lm"] = _expand_config_value(config["lm"])
    if config.get("av_enc", {}).get("path_to_pretrained"):
        config["av_enc"]["path_to_pretrained"] = _resolve_optional_path(
            config["av_enc"]["path_to_pretrained"],
            root=_resolve_data_root("${BCMLF_CHECKPOINT_ROOT}"),
        )

    config = edict(config) # use edict for backward compatibility with MMSA v1.0

    return config


def get_config_all(config_file: str) -> dict:
    """
    Get all default configs. This function is used to export default config file. 
    If you want to get config for a specific model, use "get_config_regression" or "get_config_tune" instead.

    Parameters:
        config_file: "regression" or "tune"
    
    Returns:
        config: all default configs
    """
    if config_file == "regression":
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    elif config_file == "tune":
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    else:
        raise ValueError("config_file should be 'regression' or 'tune'")
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    return edict(config_all)

def get_citations() -> dict:
    """
    Get paper titles and citations for models and datasets.

    Returns:
        cites (dict): {
            models: {
                tfn: {
                    title: "xxx",
                    paper_url: "xxx",
                    citation: "xxx",
                    description: "xxx"
                },
                ...
            },
            datasets: {
                ...
            },
        }
    """
    # TODO: add citations
    config_file = Path(__file__).parent / "config" / "citations.json"
    with open(config_file, 'r') as f:
        cites = json.load(f)
    return cites
