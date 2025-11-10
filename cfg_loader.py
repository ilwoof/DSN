import argparse
import json
import os
import sys
from typing import Dict, Any, Optional


def flatten_dict(d: Dict, remove_keys: Optional[set] = None) -> Dict:
    """Flatten a dictionary by merging sub-dicts without prefixes."""
    remove_keys = remove_keys or set()
    flat = {}
    for key, value in d.items():
        if key in remove_keys:
            continue
        if isinstance(value, dict):
            flat.update(flatten_dict(value, remove_keys))
        else:
            flat[key] = value
    return flat


def recursive_merge(base: Dict, new: Dict) -> Dict:
    """Deep merge dictionaries, with new overwriting base."""
    merged = base.copy()
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overrides_recursively(d: Dict, overrides: Dict) -> None:
    """Recursively apply overrides to leaf nodes in the dictionary."""
    for key in list(d.keys()):
        if isinstance(d[key], dict):
            apply_overrides_recursively(d[key], overrides)
        elif key in overrides and overrides[key] is not None:
            d[key] = overrides[key]


def remove_comments(d: Any) -> Any:
    """Recursively remove '_comment' keys from dictionary or list."""
    if isinstance(d, dict):
        return {k: remove_comments(v) for k, v in d.items() if k != "_comment"}
    elif isinstance(d, list):
        return [remove_comments(item) for item in d]
    return d


def load_config_from_json(json_path: Optional[str] = None, cli_overrides: Optional[Dict] = None) -> argparse.Namespace:
    """Load and merge configuration from CLI args, JSON, and defaults."""
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Malware Classifier')

    # Define CLI arguments with defaults
    parser.add_argument('--framework', type=str, choices=['DAUGL'], default='DAUGL')
    parser.add_argument('--paradigm', type=str, choices=['DAN'], default='DAN')
    parser.add_argument('--run_times', type=int, default=1)
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--root_path', type=str, default="./data/")
    parser.add_argument('--dataset_name', type=str, choices=['EMBER2024'], default='EMBER2024')
    parser.add_argument('--dataset_type', type=str, choices=['debug', 'full'], default='debug')
    parser.add_argument('--sampling_strategy', type=str, default='allsample')
    parser.add_argument('--data_split_ratio', type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument('--train_data_domain', nargs="+", type=str, choices=['win32', 'win64', 'elf', 'apk'], default=['win32', 'elf'])
    parser.add_argument('--test_data_domain', nargs="+", type=str, choices=['elf', 'win32', 'win64', 'apk'], default=['win32', 'elf'])
    parser.add_argument('--labeled_domain', nargs="+", type=str, choices=['win32', 'win64', 'apk', 'elf'], default=['win32'])
    parser.add_argument('--domain_tag', type=str, choices=['pfm_label'], default='pfm_label')
    parser.add_argument('--domain_granularity', type=str, choices=['binary'], default='binary')
    parser.add_argument('--feature_setting', type=str, choices=['elfdata'], default='elfdata')
    parser.add_argument('--feature_normalize', type=str, default='none')
    parser.add_argument('--network_setting', type=str, default='with_da_branch')
    parser.add_argument('--model_setting', type=str, default='mlp')
    parser.add_argument('--input_feature_dim', type=int, default=689)
    parser.add_argument('--feature_embedding_dim', type=int, default=32)
    parser.add_argument('--feature_hidden_dim', type=int, default=16)
    parser.add_argument('--mlp_layer_num', type=int, default=3)
    parser.add_argument('--mlp_act_func', type=str, default='relu')
    parser.add_argument('--mlp_dropout', type=float, default=0.1)
    parser.add_argument('--mlp_norm_type', type=str, default='rmsnorm')
    parser.add_argument('--classifier_type', type=str, default='moe')
    parser.add_argument('--experts_num', type=int, default=5)
    parser.add_argument('--cls_layer_num', type=int, default=2)
    parser.add_argument('--cls_norm_type', type=str, default='batchnorm')
    parser.add_argument('--cls_dropout', type=float, default=0.5)
    parser.add_argument('--cls_input_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lambda_balance', type=float, default=0.003)
    parser.add_argument('--lambda_delta', type=float, default=0.04)
    parser.add_argument('--optimizer_setting', type=str, default='adamw')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1.2e-3)
    parser.add_argument('--scheduler_setting', type=str, default='PolynomialDecayLR')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--warmup_epoch', type=int, default=3)
    parser.add_argument('--progressive_epoch', type=int, default=6,
                        help="Epoch number for lambda_val in the GRL layer from 0.0 to 1.0")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_save_path', type=str, default='./checkpoint/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--redirect_to_logfile', action='store_true', default=True)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--evaluation_mode', type=str, default='valid_set')
    parser.add_argument('--monitor_metric', type=str, default='valid_loss')

    # Differential network and prototype setting
    parser.add_argument('--diff_module', action='store_true', default=True)
    parser.add_argument('--prototypes_per_set', type=int, default=5)
    parser.add_argument('--prototype_init', type=str, choices=['orthogonal'], default='orthogonal')

    # for MoE usage
    parser.add_argument('--routing_type', type=str, choices=['diff_only'], default='diff_only')
    parser.add_argument('--repres_type', type=str,  choices=['raw_feature'], default='raw_feature')

    parser.add_argument('--diff_loss', type=str, default='simplified')
    parser.add_argument('--delta_scope', type=str, choices=['source_only'], default='source_only')
    parser.add_argument('--per_sample_delta', action='store_true', default=False)
    parser.add_argument('--visualization', action='store_true', default=False)

    # Parse CLI arguments
    args, _ = parser.parse_known_args()
    cli_args = vars(args)

    # Detect explicitly passed CLI args (ignore defaults)
    explicit_cli_args = {k: cli_args[k] for k in cli_args if f"--{k}" in sys.argv or k in (cli_overrides or {})}
    if cli_overrides:
        explicit_cli_args.update(cli_overrides)

    # CLI defaults for fallback
    cli_defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}

    # Determine JSON path
    Framework2Config = {'DAUGL': 'dan'}
    if json_path is None:
        framework_key = explicit_cli_args.get("framework", cli_defaults.get("framework"))
        json_path = os.path.join(os.getcwd(),
                                 f"src/preset_configs/{Framework2Config.get(framework_key, 'dan')}_config.json")

    # Load and clean JSON configuration
    try:
        with open(json_path, 'r') as f:
            raw_json = json.load(f)
        raw_json = remove_comments(raw_json)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not read JSON at {json_path}: {e}. Using CLI defaults/overrides.")
        return argparse.Namespace(**cli_args)

    # Flatten top-level JSON sections
    flat_sections = {}
    for section in ["general", "training", "output", "logging"]:
        if section in raw_json:
            flat_sections.update(flatten_dict(raw_json[section]))

    # Flatten selected learning rate strategy
    lr_strategy_name = explicit_cli_args.get("scheduler_setting", cli_defaults.get("scheduler_setting"))
    try:
        strategy_dict = raw_json.get("training", {}).get("scheduler_setting", {}).get(lr_strategy_name, {})
        flat_sections.update(flatten_dict(strategy_dict))
    except (KeyError, TypeError):
        pass


    # Merge configurations: CLI defaults < JSON flat < CLI explicit
    merged_config = cli_defaults.copy()
    merged_config.update(flat_sections)
    merged_config.update(explicit_cli_args)

    if "selected_dataset" in merged_config:
        merged_config["dataset_name"] = merged_config.pop("selected_dataset")

    return argparse.Namespace(**merged_config)



if __name__ == "__main__":
    args = load_config_from_json()
    for key, value in vars(args).items():
        print(f"{key}: {value}")