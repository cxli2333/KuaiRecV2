# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import re
import sys
import yaml
import json
import hashlib
import datetime
import tiktoken
from copy import deepcopy
from typing import Union, List, Optional, Dict, Any
import logging
from logging import getLogger

from model.model import AbstractModel
from dataset.dataset import AbstractDataset, SeqAugDataset
from accelerate.utils import set_seed

def _ensure_suffix(name: str, suffix: str = ".pkl") -> str:
    name = str(name)
    return name if name.endswith(suffix) else f"{name}{suffix}"

def _append_tag(base: str, tag: str | None) -> str:
    if not tag or tag == "None":
        return base
    # Sanitize: retain only alphanumeric characters, underscores, hyphens, and dots
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag).strip())
    return f"{base}_{safe}" if safe else base

def _require_mapping(d: dict, key: str):
    if key not in d:
        raise KeyError(f"Missing required key '{key}' in config mapping.")

def build_dataset_config(final_config: dict, dataset_name: str) -> dict:
    """Build or normalize configuration from final_config[dataset_name] and return final_config."""
    data_file_cfg = final_config.get(dataset_name, {})
    if not isinstance(data_file_cfg, dict):
        raise ValueError(
            f"Config for dataset '{dataset_name}' must be a mapping, got: {type(data_file_cfg)}"
        )

    # ---- Check required fields when mandatory ----
    _require_mapping(data_file_cfg, "data_path_prefix")
    _require_mapping(data_file_cfg, "raw_seq_name")
    _require_mapping(data_file_cfg, "item2codes_name")

    prefix: str = data_file_cfg.get("data_path_prefix")
    raw_seq_name: str = data_file_cfg.get("raw_seq_name")
    item2codes_name: str = data_file_cfg.get("item2codes_name")

    # Optionally copy numeric parameters into final_config
    for k in ("n_digits", "codebook_size"):
        if k in data_file_cfg:
            final_config[k] = data_file_cfg[k]

    # Append tags only when provided
    sent_emb_model_tag = data_file_cfg.get("sent_emb_model_tag")
    quantizer_model_tag = data_file_cfg.get("Quantizer_model_tag")
    item2codes_name = _append_tag(item2codes_name, sent_emb_model_tag)
    item2codes_name = _append_tag(item2codes_name, quantizer_model_tag)

    # Normalize the .pkl suffix
    raw_seq_name = _ensure_suffix(raw_seq_name, ".pkl")
    item2codes_name = _ensure_suffix(item2codes_name, ".pkl")

    # Write back filenames without directory components
    #final_config["raw_seq_name"] = raw_seq_name
    #final_config["item2codes_name"] = item2codes_name
    
    use_category_subdir = bool(final_config.get("use_category_subdir", False))
    category = final_config.get("category", dataset_name)
    if use_category_subdir:
        final_config['dataset'] = category
        raw_seq_path = os.path.join(prefix, category, "processed", raw_seq_name)
        item2codes_path = os.path.join(prefix, category, "processed", item2codes_name)
    else:
        final_config['dataset'] = dataset_name
        raw_seq_path = os.path.join(prefix, dataset_name, raw_seq_name)
        item2codes_path = os.path.join(prefix, dataset_name, item2codes_name)

    final_config["raw_seq_path"] = raw_seq_path
    final_config["item2codes_path"] = item2codes_path
    return final_config

def build_model_config(config: Dict) -> Dict:
    """
    Post-process the common config:
    - For T5, fill model hyperparameters from presets when presets and model_size exist.
        Override priority: config_dict > config_file > model/dataset/root config > preset.

    Args:
        config: merged configuration after combining YAML and dictionary inputs in get_config
        model_name: model name

    Returns:
        dict: processed configuration, modified in place and returned
    """
    logger = logging.getLogger(__name__)

    # 1. Optional T5 preset expansion
    # Convention: if any config.yaml / config_dict provides:
    #   - model_size: "small"
    #   - presets: { tiny: {...}, small: {...}, ... }
    # Read default hyperparameters from presets[model_size],
    # and fill only fields not already present in config.
    # This preserves the original field access pattern:
    #   num_layers = config["num_layers"]
    #   d_model    = config["d_model"]
    # and so on.
    # It also preserves the override priority.
    if "presets" in config and "model_size" in config:
        presets = config.get("presets") or {}
        model_size = str(config.get("model_size")).strip()

        if model_size not in presets:
            logger.warning(
                "[build_dataset_config] model_size=%r not found in presets keys=%r; "
                "skip T5 preset expansion.",
                model_size, list(presets.keys())
            )
        else:
            preset_cfg = deepcopy(presets[model_size])
            
            t5_keys = preset_cfg.keys()

            for k in t5_keys:
                config[k] = preset_cfg[k]

            logger.info(
                "[build_dataset_config] Applied T5 preset '%s', with overrides kept from config.",
                model_size
            )

        # Remove presets here if they should not appear in the final config or logs:
        config.pop("presets", None)
    return config

def init_seed(seed, reproducibility=True):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def get_command_line_args_str():
    return '_'.join(sys.argv).replace('/', '|')


def get_file_name(config: dict, suffix: str = ''):
    # Retain only time and dataset information to keep filenames short
    dataset_name = config.get("dataset", "unknown")
    # Extract the timestamp from run_local_time
    time_str = config.get('run_local_time', 'unknown')
    
    # Generate a short filename: dataset_timestamp
    # Use the timestamp directly when available
    if time_str != 'unknown':
        logfilename = "{}_{}{}".format(dataset_name, time_str, suffix)
    else:
        # Use the current time when no timestamp is available
        current_time = get_local_time()
        logfilename = "{}_{}{}".format(dataset_name, current_time, suffix)
    
    return logfilename


def init_logger(config: dict):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)
    model_name = os.path.join(dataset_name, config["model"])
    os.makedirs(model_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log')
    if len(logfilename) > 255:
        logfilename = logfilename[:245] + logfilename[-10:]
    logfilepath = os.path.join(LOGROOT, config["dataset"], config["model"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])

    if not config['accelerator'].is_main_process:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()

def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


def get_tokenizer(model_name: str):
    """
    Retrieves the tokenizer for a given model name.

    Args:
        model_name (str): The model name.

    Returns:
        AbstractTokenizer: The tokenizer for the given model name.

    Raises:
        ValueError: If the tokenizer is not found.
    """
    try:
        tokenizer_class = getattr(
            importlib.import_module(f'genrec.models.{model_name}.tokenizer'),
            f'{model_name}Tokenizer'
        )
    except:
        raise ValueError(f'Tokenizer for model "{model_name}" not found.')
    return tokenizer_class


def get_model(model_name: Union[str, AbstractModel]) -> AbstractModel:
    """
    Retrieves the model class based on the provided model name.

    Args:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the model class.

    Returns:
        AbstractModel: The model class corresponding to the provided model name.

    Raises:
        ValueError: If the model name is not found.
    """
    if isinstance(model_name, AbstractModel):
        return model_name

    try:
        model_class = getattr(
            importlib.import_module('model'),
            model_name
        )
    except:
        raise ValueError(f'Model "{model_name}" not found.')
    return model_class

# If this function shares a file with SeqAugDataset/AbstractDataset, ensure these imports are active
# from .dataset import AbstractDataset, SeqAugDataset
# Classes in the same module can also be resolved using getattr(current_module, ...)

def get_dataset(dataset_name: Union[str, Any], config: dict):
    """
    Choose the data split provider based on the parameters:
      - data_split_mode = "user_split": use classes in the current module, e.g. SeqAugDataset
      - data_split_mode = "leave_one_out": use classes from genrec.datasets
    Support the legacy use_user_split=True/False setting.

    Args:
        dataset_name: class name string or an existing dataset instance
        config: global configuration containing data_split_mode or use_user_split

    Returns:
        An uninstantiated dataset class, or the supplied instance
    """
    # Return an existing instance directly
    try:
        from .dataset import AbstractDataset  # A direct reference is also possible if the function and class share a file
    except Exception:
        # Use an absolute import externally if relative imports are inconvenient
        pass

    if hasattr(dataset_name, '__class__') and \
       dataset_name.__class__.__name__ not in ('str', 'unicode', 'bytes'):
        # Heuristic: a non-string value that appears to be an instance
        return dataset_name

    # Resolve data_split_mode / use_user_split
    mode = str(config.get('data_split_mode', '')).lower().strip()
    if not mode:
        use_user_split = config.get('use_user_split', False)
        mode = 'user_split' if (isinstance(use_user_split, bool) and use_user_split) or \
                              (isinstance(use_user_split, str) and use_user_split.lower() == 'true') \
               else 'leave_one_out'

    # Normalize strings
    if isinstance(dataset_name, bytes):
        dataset_name = dataset_name.decode('utf-8')
    if not isinstance(dataset_name, str):
        raise ValueError(f'Invalid dataset_name type: {type(dataset_name)}')

    # Support aliases in user_split mode by mapping alternative names to SeqAugDataset
    alias_local = {
        'user_split': ['SeqAugDataset', 'UserSplitDataset', 'UserSplit', 'UserSeq'],
    }

    def _resolve_from(module_name: str, cls_name: str):
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, cls_name)
        except Exception as e:
            raise ValueError(f'Cannot resolve "{cls_name}" from module "{module_name}": {e}')

    if mode == 'user_split':
        # Resolve from the current module, e.g. SeqAugDataset
        current_module = importlib.import_module(__name__)

        # Try the SeqAugDataset alias if dataset_name is not in the current module
        if not hasattr(current_module, dataset_name):
            # Fall back to the local default class name: SeqAugDataset
            dataset_name = 'SeqAugDataset'
        try:
            return getattr(current_module, dataset_name)
        except AttributeError:
            raise ValueError(f'Dataset class "{dataset_name}" not found in local module {__name__} '
                             f'(user_split mode). Make sure SeqAugDataset is defined.')

    elif mode == 'leave_one_out':
        # Resolve from genrec.datasets, retaining the original logic
        try:
            return _resolve_from('genrec.datasets', dataset_name)
        except ValueError:
            # Allow fallback handling when the default LOO dataset name was intended
            raise

    else:
        raise ValueError(f'Unknown data_split_mode: {mode}. '
                         f'Expect "leave_one_out" or "user_split".')


def get_trainer(model_name: Union[str, AbstractModel]):
    """
    Returns the trainer class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        trainer_class: The trainer class corresponding to the given model name. If the model name is not found, the default Trainer class is returned.
    """
    from trainer.trainer import Trainer
    if isinstance(model_name, str):
        try:
            trainer_class = getattr(
                importlib.import_module(f'trainer.models.{model_name}.trainer'),
                f'{model_name}Trainer'
            )
            return trainer_class
        except:
            return Trainer
    else:
        return Trainer


def get_pipeline(model_name: Union[str, AbstractModel]):
    """
    Returns the pipeline class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        pipeline_class: The pipeline class corresponding to the given model name. If the model name is not found, the default Pipeline class is returned.
    """
    from genrec.pipeline import Pipeline
    if isinstance(model_name, str):
        try:
            pipeline_class = getattr(
                importlib.import_module(f'genrec.models.{model_name}.pipeline'),
                f'{model_name}Pipeline'
            )
            return pipeline_class
        except:
            return Pipeline
    else:
        return Pipeline

def get_total_steps(config, train_dataloader):
    """
    Calculate the total number of steps for training based on the given configuration and dataloader.

    Args:
        config (dict): The configuration dictionary containing the training parameters.
        train_dataloader (DataLoader): The dataloader for the training dataset.

    Returns:
        int: The total number of steps for training.

    """
    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']

def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        
        # Use safe parsing instead of eval
        try:
            # Check boolean values first
            if v.lower() in ['true', 'false']:
                config[key] = (v.lower() == 'true')
                continue
            
            # Try JSON parsing first; it supports nested values safely
            new_v = json.loads(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except Exception:
            try:
                # Then try ast.literal_eval for literals only
                import ast
                new_v = ast.literal_eval(v)
                if new_v is not None and not isinstance(
                    new_v, (str, int, float, bool, list, dict, tuple)
                ):
                    new_v = v
            except Exception:
                # Keep the original string if both parsers fail
                new_v = v
        
        config[key] = new_v
    return config

def get_config(
    model_name: str,
    dataset_name: str,
    category: Optional[str],
    config_file: Optional[Union[str, List[str]]],
    config_dict: Optional[dict]
) -> dict:
    """
    Get the configuration for a model and dataset.
    Overwrite rule: config_dict > config_file > model config.yaml > dataset config.yaml > config.yaml

    Args:
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.
        category (str): The name of the category in Amazon Dataset.
        config_file (Union[str, list[str], None]): The path to additional configuration file(s) or a list of paths to multiple additional configuration files. If None, default configurations will be used.
        config_dict (Optional[dict]): A dictionary containing additional configuration options. These options will override the ones loaded from the configuration file(s).
    Returns:
        dict: The final configuration dictionary.
    Raises:
        FileNotFoundError: If any of the specified configuration files cannot be found.
    Note:
        - If `model_name` is a string, the function will attempt to load the model's configuration file located at `./model/{model_name}/config.yaml`.
        - If `dataset_name` is a string, the function will attempt to load the dataset's configuration file located at `./dataset/config.yaml`.
        - The function will merge the configurations from all the specified configuration files and the `config_dict` parameter.
    """
    final_config = {}
    logger = getLogger()
    
    # 1. Load default configs
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file_list = [os.path.join(current_path, 'config.yaml')]
    # 2. Load dataset specific configs
    if isinstance(dataset_name, str):
        config_file_list.append(
            os.path.join(current_path, f'dataset/config.yaml')
        )
    # 3. Load model specific configs
    if isinstance(model_name, str):
        config_file_list.append(
            os.path.join(current_path, f'model/{model_name}/config.yaml')
        )
        final_config['model'] = model_name
    # 4. Load file specific configs
    if config_file:
        if isinstance(config_file, str):
            config_file = [config_file]
        config_file_list.extend(config_file)
    for file in config_file_list:
        cur_config = yaml.safe_load(open(file, 'r'))
        if cur_config is not None:
            final_config.update(cur_config)
    # 5. Load dict configs
    if config_dict:
        final_config.update(config_dict)
    if "Amazon" in dataset_name:
        final_config["use_category_subdir"] = bool(True)
        final_config["category"] = category
        
    final_config = build_dataset_config(final_config, dataset_name)
    final_config = build_model_config(final_config)
    
    final_config['run_local_time'] = get_local_time()
    final_config = convert_config_dict(final_config)
    
    logger.info("Final config:\n%s", json.dumps(final_config, indent=2, ensure_ascii=False))
    return final_config


def parse_command_line_args(unparsed: List[str]) -> Dict[str, Any]:
    """
    Parses command line arguments and returns a dictionary of key-value pairs.

    Args:
        unparsed (list[str]): A list of command line arguments in the format '--key=value'.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.

    Example:
        >>> parse_command_line_args(['--name=John', '--age=25', '--is_student=True'])
        {'name': 'John', 'age': 25, 'is_student': True}
    """
    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=', 1)  # Split only on the first = to preserve = characters inside dictionaries
        key = key[len('--'):]
        
        # Use safe parsing instead of eval
        parsed_value = value
        try:
            # Try JSON parsing first; it supports nested values safely
            parsed_value = json.loads(value)
        except Exception:
            try:
                # Then try ast.literal_eval for literals only
                import ast
                parsed_value = ast.literal_eval(value)
            except Exception:
                # Keep the original string if both parsers fail
                pass
        
        args[key] = parsed_value
    return args

def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp

def config_for_log(config: dict) -> dict:
    config = config.copy()
    # Remove non-serializable or irrelevant objects
    config.pop('device', None)
    config.pop('accelerator', None)

    # TensorBoard hparams accepts only int, float, str, bool, and torch.Tensor
    # Convert non-scalar values to strings to avoid type errors
    sanitized = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)):
            sanitized[k] = v
        else:
            # Convert list/tuple/dict/None/numpy values and other types to strings
            try:
                sanitized[k] = str(v)
            except Exception:
                sanitized[k] = '<unserializable>'
    return sanitized


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
