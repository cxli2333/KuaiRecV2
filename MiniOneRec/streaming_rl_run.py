import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
import random
import torch
import sys
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig
import re

from minionerec_trainer import ReReTrainer

SID_LEVEL_PATTERN = re.compile(r"<[a-zA-Z]_[^>]+>")

prompt2target = {}
prompt2history = {}
history2target = {}
prompt2behaviors = {}

# 1. Use the same SID parsing logic as create_master_files.py
def parse_sid_list(sid_str):
    """
    Convert a raw sid_list such as "417,224,387"
    to the SID token string used for training:
    "<sid_start_flag><a_417><b_224><c_387><sid_end_flag>"
    """
    if not isinstance(sid_str, str) or not sid_str:
        return None, None

    raw_tokens = sid_str.split(',')
    prefixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    token_list = ["<sid_start_flag>"]
    for i, t in enumerate(raw_tokens):
        t = t.strip()
        if not t:
            continue
        if i < len(prefixes):
            prefix = prefixes[i]
        else:
            prefix = f"l{i}"
        token_list.append(f"<{prefix}_{t}>")
    token_list.append("<sid_end_flag>")

    sid_concat = "".join(token_list)
    return token_list, sid_concat


# 2. Basic utility functions
def safe_strip(x):
    if x is None:
        return None
    return str(x).strip()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_history_cell(x):
    """
    Compatible formats:
    - "['sid1', 'sid2']"
    - ['sid1', 'sid2']
    - "sid1"
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]

    x = str(x).strip()
    if not x:
        return []

    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(v).strip() for v in val if str(v).strip()]
    except Exception:
        pass

    return [x]

def build_subtree_items(sid_rows, sid2emb, target_sids, subtree_sample_ratio=0.2, seed=42):
    """
    Build a smaller subtree:
    - Retain all target items
    - Randomly sample the remaining non-target items by the specified ratio
    """
    rng = random.Random(seed)

    all_sid_set = {s for s, _, _ in sid_rows}
    target_sids = set([s for s in target_sids if s in all_sid_set])
    non_target_sids = sorted(list(all_sid_set - target_sids))

    sample_num = int(len(non_target_sids) * subtree_sample_ratio)
    sample_num = min(sample_num, len(non_target_sids))

    sampled_non_target = set(rng.sample(non_target_sids, sample_num)) if sample_num > 0 else set()
    selected_sids = target_sids | sampled_non_target

    selected_rows = [(s, t, i) for s, t, i in sid_rows if s in selected_sids]
    selected_sid2emb = {s: e for s, e in sid2emb.items() if s in selected_sids}

    print(
        f"[Subtree] targets={len(target_sids)}, sampled_non_targets={len(sampled_non_target)}, "
        f"total={len(selected_sids)} / full={len(all_sid_set)}"
    )

    return selected_sid2emb, selected_rows

def build_prompt(history_sids, history_len=20):
    """
    Keep this consistent with the SFT single-sample format where possible.
    Adjust this function if StreamingSid2SidDataset uses a slightly different prompt.
    """
    history_sids = [str(x).strip() for x in history_sids if str(x).strip()]
    history_sids = history_sids[-history_len:]
    history_str = "".join(history_sids)

    prompt_input = (
        f"The user has interacted with items {history_str} in chronological order. "
        f"Can you predict the next possible item that the user may expect?"
    )
    prompt = f"### User Input: \n{prompt_input}\n\n### Response:\n"
    return prompt, history_str


def parse_emb_vec(x):
    """
    Parse emb_vec:
    - "1.0,2.0,3.0"
    - "[1.0,2.0,3.0]"
    """
    if x is None:
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=np.float32)
        return arr

    x = str(x).strip()
    if not x:
        return None

    if x.startswith("[") and x.endswith("]"):
        x = x[1:-1]

    parts = [p.strip() for p in x.split(",") if p.strip()]
    if len(parts) == 0:
        return None

    try:
        arr = np.asarray([float(v) for v in parts], dtype=np.float32)
    except Exception:
        return None

    return arr


def l2_normalize(arr, eps=1e-12):
    norm = np.linalg.norm(arr)
    if norm < eps:
        return arr
    return arr / norm


def cosine_sim(a, b, eps=1e-12):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb + eps))


def normalize_sid_completion(text):
    """
    The trainer passes decoded completion strings to the reward function.
    Apply only lightweight cleanup to preserve the SID token structure.
    """
    if text is None:
        return None

    text = str(text).strip()
    text = text.replace("\n", "")
    text = text.strip('"').strip("'")

    # Keep only the content through sid_end_flag
    end_flag = "<sid_end_flag>"
    idx = text.find(end_flag)
    if idx != -1:
        text = text[: idx + len(end_flag)]

    return text.strip()


# 3. Load candidate SIDs and embeddings from item_files
def load_item_semantic_space(item_files):
    """
    Input: item_files with columns sid_list, emb_vec, item_id, tag_list
    Output:
    - sid2emb: training-space sid_str -> embedding
    - sid2item: sid_str -> item_id
    - sid_rows: records used to generate info_file
    """
    sid2emb = {}
    sid2item = {}
    sid_rows = []

    col_names = ["sid_list", "emb_vec", "item_id", "tag_list"]

    for fp in item_files:
        fp = fp.strip()
        if not fp:
            continue

        if not os.path.exists(fp):
            print(f"[Warning] item file not found: {fp}")
            continue

        print(f"[Load Item File] {fp}")

        df = pd.read_csv(
            fp,
            sep="\t",
            header=None,
            names=col_names,
            dtype={"item_id": str}
        )

        for _, row in df.iterrows():
            raw_sid = row["sid_list"]
            emb_vec = row["emb_vec"]
            item_id = safe_strip(row["item_id"])

            _, sid_str = parse_sid_list(raw_sid)
            emb = parse_emb_vec(emb_vec)

            if sid_str is None:
                continue
            if emb is None or len(emb) == 0:
                continue

            emb = l2_normalize(emb)

            sid2emb[sid_str] = emb
            sid2item[sid_str] = item_id

    # Used to write info_file
    for sid_str, item_id in sid2item.items():
        # The trainer reads the second column; use item_id as a placeholder when titles are unavailable.
        sid_rows.append((sid_str, f"item_{item_id}", item_id))

    print(f"[Semantic Space] valid sid count = {len(sid2emb)}")
    return sid2emb, sid2item, sid_rows


def ensure_info_file(sid_rows, output_dir):
    """
    The trainer requires info_file and builds prefix constraints from its first semantic_id column.
    File format:
        semantic_id \t item_title \t item_id
    """
    os.makedirs(output_dir, exist_ok=True)
    info_file = os.path.join(output_dir, "info_file.tsv")

    with open(info_file, "w", encoding="utf-8") as f:
        for sid_str, title, item_id in sid_rows:
            f.write(f"{sid_str}\t{title}\t{item_id}\n")

    print(f"[Info File] saved to: {info_file}")
    return info_file


# 4. Build the training dataset from RL train.tsv
def build_rl_train_dataset(
    train_file,
    positive_behavior_cols=None,
    interaction_behavior_cols=None,
    negative_behavior_cols=None,
    history_len=20,
    seed=42
):
    """
    Read RL training data.

    Expected train_file columns include:
        user_id, history, target

    Read behavior columns dynamically from positive_behavior_cols / negative_behavior_cols.
    Fill missing behavior columns with zero.
    """

    positive_behavior_cols = positive_behavior_cols or []
    interaction_behavior_cols =interaction_behavior_cols or []
    negative_behavior_cols = negative_behavior_cols or []
    behavior_cols = list(dict.fromkeys(positive_behavior_cols + interaction_behavior_cols + negative_behavior_cols))

    # Prefer reading the header
    df = pd.read_csv(train_file, sep="\t", dtype=str)

    required_cols = ["user_id", "history", "target"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"train_file must contain column '{col}', "
                f"but only got columns: {list(df.columns)}"
            )

    # Keep only behavior columns that actually exist
    existing_behavior_cols = [c for c in behavior_cols if c in df.columns]
    missing_behavior_cols = [c for c in behavior_cols if c not in df.columns]

    if len(missing_behavior_cols) > 0:
        print(f"[Train Dataset] missing behavior cols, will fill 0: {missing_behavior_cols}")

    prompts = []
    prompt2target = {}
    prompt2history = {}
    history2target = {}
    prompt2behaviors = {}

    for _, row in df.iterrows():
        history_sids = parse_history_cell(row["history"])
        target_sid = safe_strip(row["target"])

        if len(history_sids) == 0 or not target_sid:
            continue

        prompt, history_str = build_prompt(history_sids, history_len=history_len)

        prompts.append(prompt)
        prompt2target[prompt] = target_sid
        prompt2history[prompt] = history_str
        history2target[history_str] = target_sid

        behavior_info = {}
        for c in behavior_cols:
            if c in existing_behavior_cols:
                val = row.get(c, 0)
                try:
                    behavior_info[c] = float(val) if val is not None and str(val).strip() != "" else 0.0
                except Exception:
                    behavior_info[c] = 0.0
            else:
                behavior_info[c] = 0.0

        prompt2behaviors[prompt] = behavior_info

    train_dataset = Dataset.from_dict({"prompt": prompts}).shuffle(seed=seed)

    print(f"[Train Dataset] num prompts = {len(prompts)}")
    print(f"[Train Dataset] positive_behavior_cols = {positive_behavior_cols}")
    print(f"[Train Dataset] interaction_behavior_cols = {interaction_behavior_cols}")
    print(f"[Train Dataset] negative_behavior_cols = {negative_behavior_cols}")
    print(f"[Train Dataset] existing_behavior_cols = {existing_behavior_cols}")

    return train_dataset, prompt2target, prompt2history, history2target, prompt2behaviors


# 5. Build the evaluation dataset from test_T.csv
def build_eval_dataset(eval_file, history_len=20, seed=42):
    if eval_file is None or not os.path.exists(eval_file):
        print("[Eval Dataset] eval file not found, skip.")
        return None, {}, {}, {}

    df = pd.read_csv(eval_file)

    prompts = []
    prompt2target = {}
    prompt2history = {}
    history2target = {}

    for _, row in df.iterrows():
        history_sids = parse_history_cell(row["history_item_sid"])
        target_sid = safe_strip(row["item_sid"])

        if len(history_sids) == 0 or not target_sid:
            continue

        prompt, history_str = build_prompt(history_sids, history_len=history_len)

        prompts.append(prompt)
        prompt2target[prompt] = target_sid
        prompt2history[prompt] = history_str
        history2target[history_str] = target_sid

    eval_dataset = Dataset.from_dict({"prompt": prompts}).shuffle(seed=seed)
    print(f"[Eval Dataset] num prompts = {len(prompts)}")
    return eval_dataset, prompt2target, prompt2history, history2target


# 6. main
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--item_files", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_len", type=int, default=20)

    # Training arguments, aligned with the original rl.py where possible
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--add_gt", action="store_true")
    parser.add_argument("--eval_step", type=float, default=0.2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--test_during_training", action="store_true")
    parser.add_argument("--dynamic_sampling", action="store_true")
    parser.add_argument("--sync_ref_model", action="store_true")
    parser.add_argument("--test_beam", type=int, default=20)
    parser.add_argument("--dapo", action="store_true")
    parser.add_argument("--gspo", action="store_true")

    # Required by the trainer but not natively supported by the referenced trl==0.12.1 GRPOConfig
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=64)
    parser.add_argument(
        "--reward_type",
        type=str,
        default="exact_prefix_semantic",
        choices=[
            "exact",
            "prefix",
            "semantic",
            "exact_prefix_semantic",
            "exact_prefix_behavior",
        ]
    )

    parser.add_argument("--w_exact", type=float, default=1.0)
    parser.add_argument("--w_prefix", type=float, default=1.0)
    parser.add_argument("--w_sem", type=float, default=0.2)

    parser.add_argument("--subtree_sample_ratio", type=float, default=0.1)

    parser.add_argument("--w_pos", type=float, default=1)
    parser.add_argument("--w_inter",type=float, default =0.5)
    parser.add_argument("--w_neg", type=float, default=0.5)
    
    parser.add_argument(
        "--positive_behavior_cols",
        nargs="+",
        default=[
            "is_long_view",
        ]
    )
    parser.add_argument(
        "--interaction_behavior_cols",
        nargs="+",
        default=[
            "is_like",
            "is_follow",
            "is_forward",
            "is_comment",
            "is_search",
            "is_profile_enter",
        ]
    )
    parser.add_argument(
        "--negative_behavior_cols",
        nargs="+",
        default=["is_hate"]
    )


    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer from the SFT or previous RL checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 1) Build the semantic space and info_file
    item_files = [x for x in args.item_files.split(",") if x.strip()]
    sid2emb, _ , sid_rows = load_item_semantic_space(item_files)
    

    # 2) Build training and evaluation datasets
    train_dataset, train_prompt2target, train_prompt2history, train_history2target, train_prompt2behaviors = build_rl_train_dataset(
        train_file=args.train_file,
        positive_behavior_cols=args.positive_behavior_cols,
        interaction_behavior_cols = args.interaction_behavior_cols,
        negative_behavior_cols=args.negative_behavior_cols,
        history_len=args.history_len,
        seed=args.seed
    )


    eval_dataset, eval_prompt2target ,eval_prompt2history, eval_history2target = build_eval_dataset(
        eval_file=args.eval_file,
        history_len=args.history_len,
        seed=args.seed
    )
    all_target_sids = set(train_prompt2target.values()) | set(eval_prompt2target.values())

    if 0.0 < args.subtree_sample_ratio < 1.0:
        sid2emb, sid_rows = build_subtree_items(
            sid_rows=sid_rows,
            sid2emb=sid2emb,
            target_sids=all_target_sids,
            subtree_sample_ratio=args.subtree_sample_ratio,
            seed=args.seed,
        )

    info_file = ensure_info_file(sid_rows, args.output_dir)

    prompt2target = {}
    prompt2target.update(train_prompt2target)
    prompt2target.update(eval_prompt2target)

    prompt2history = {}
    prompt2history.update(train_prompt2history)
    prompt2history.update(eval_prompt2history)

    history2target = {}
    history2target.update(train_history2target)
    history2target.update(eval_history2target)

    prompt2behaviors = {}
    prompt2behaviors.update(train_prompt2behaviors)

    # 3) sanity check
    train_targets = list(train_prompt2target.values())
    hit_count = sum([1 for t in train_targets if t in sid2emb])
    print(f"[Sanity] train target hit sid2emb = {hit_count}/{len(train_targets)}")

    def extract_sid_levels(sid: str):
        """
        Extract hierarchical tokens from a SID, for example:
        <sid_start_flag><a_123><b_456><c_789><sid_end_flag>
        -> ['<a_123>', '<b_456>', '<c_789>']
        """
        if sid is None:
            return []
        sid = str(sid).strip()
        return SID_LEVEL_PATTERN.findall(sid)
    
    def behavior_bonus_single(prompt: str) -> float:
        behavior_info = prompt2behaviors.get(prompt, {})

        pos_cnt = 0
        inter_cnt = 0
        neg_cnt = 0
        long_view_cnt = 0

        # Handle long_view separately
        for c in args.positive_behavior_cols:
            try:
                if float(behavior_info.get(c, 0.0)) > 0:
                    pos_cnt += 1
            except Exception:
                pass

        # Handle other positive behaviors
        for c in args.interaction_behavior_cols:
            try:
                if float(behavior_info.get(c, 0.0)) > 0:
                    pos_cnt += 1
            except Exception:
                pass

        # Handle negative behaviors
        for c in args.negative_behavior_cols:
            try:
                if float(behavior_info.get(c, 0.0)) > 0:
                    neg_cnt += 1
            except Exception:
                pass

        return float(args.w_inter * inter_cnt + args.w_pos * pos_cnt - args.w_neg * neg_cnt)


    def exact_hit_reward_single(pred: str, tgt: str) -> float:
        return 1.0 if pred == tgt else 0.0

    def exact_hit_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r = exact_hit_reward_single(pred, tgt)
            rewards.append(r)

        return rewards
    
    def prefix_reward_single(pred: str, tgt: str) -> float:
        """
        Fraction of consecutive matching prefix levels.
        For an eight-level target with three initial matching prediction levels, reward = 3/8.
        """
        pred_levels = extract_sid_levels(pred)
        tgt_levels = extract_sid_levels(tgt)

        if len(pred_levels) == 0 or len(tgt_levels) == 0:
            return 0.0

        total = len(tgt_levels)
        matched = 0
        for p, t in zip(pred_levels, tgt_levels):
            if p == t:
                matched += 1
            else:
                break

        return float(matched / total)

    def semantic_reward_single(pred: str, tgt: str, sid2emb: dict) -> float:
        """
        Semantic similarity reward using the existing semantic reward logic.
        """
        if pred not in sid2emb or tgt not in sid2emb:
            return 0.0

        sim = cosine_sim(sid2emb[pred], sid2emb[tgt])
        sim = max(0.0, float(sim))
        return sim

    # 4) Reward functions, following the original rl.py style
    def exact_hit_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r = exact_hit_reward_single(pred, tgt)
            rewards.append(r)

        return rewards

    def prefix_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r = prefix_reward_single(pred, tgt)
            rewards.append(r)

        return rewards

    def semantic_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r = semantic_reward_single(pred, tgt, sid2emb)
            rewards.append(r)
        return rewards

    def exact_prefix_semantic_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        w_exact = args.w_exact
        w_prefix = args.w_prefix
        w_sem = args.w_sem

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r_exact = exact_hit_reward_single(pred, tgt)
            r_prefix = prefix_reward_single(pred, tgt)
            r_sem = semantic_reward_single(pred, tgt, sid2emb)

            total = w_exact * r_exact + w_prefix * r_prefix + w_sem * r_sem
            rewards.append(float(total))

        return rewards
    
    def exact_prefix_behavior_reward(prompts, completions, **kwargs):
        targets = [prompt2target[p] for p in prompts]
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_completion(completion)
            tgt = targets[i]

            r_exact = exact_hit_reward_single(pred, tgt)
            r_prefix = prefix_reward_single(pred, tgt)

            total = args.w_exact * r_exact + args.w_prefix * r_prefix

            # Apply behavior bonuses and penalties only on an exact hit
            if r_exact > 0:
                total += behavior_bonus_single(prompts[i])

            rewards.append(float(total))

        return rewards


    if args.reward_type == "exact":
        reward_funcs = [exact_hit_reward]

    elif args.reward_type == "prefix":
        reward_funcs = [prefix_reward]

    elif args.reward_type == "semantic":
        reward_funcs = [semantic_reward]

    elif args.reward_type == "exact_prefix_semantic":
        reward_funcs = [exact_prefix_semantic_reward]
        
    elif args.reward_type == "exact_prefix_behavior":
        reward_funcs = [exact_prefix_behavior_reward]

    else:
        raise ValueError(f"Unknown reward_type: {args.reward_type}")

    # 5) Training configuration, following the original rl.py where possible
    # Note: the referenced trl==0.12.1 GRPOConfig lacks some fields; set them in two stages.
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        save_steps=0.2,
        save_total_limit=20,
        eval_strategy="steps" if args.test_during_training else "no",
        num_generations=args.num_generations,
        temperature=args.temperature,
        sync_ref_model=args.sync_ref_model,
        per_device_eval_batch_size=args.eval_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_step,
        logging_steps=1,
        learning_rate=args.learning_rate,
        beta=args.beta,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to="wandb",
        run_name="streaming_rl",
    )

    # Manually add attributes required by minionerec_trainer.py but not native to this GRPOConfig
    training_args.max_prompt_length = args.max_prompt_length
    training_args.max_completion_length = args.max_completion_length
    training_args.remove_unused_columns = False
    training_args.seed = args.seed
    training_args.use_vllm = False
    training_args.log_completions = False
    training_args.ds3_gather_for_generation = False

    # 6) Trainer, aligned with the original rl.py where possible
    trainer = ReReTrainer(
        model=args.model_path,
        base_model=args.model_path,
        dapo=args.dapo,
        gspo=args.gspo,
        add_gt=args.add_gt,
        dynamic_sampling=args.dynamic_sampling,
        beam_search=args.beam_search,
        test_during_training=args.test_during_training,
        test_beam=args.test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    if os.environ.get("KUAIRECV2_LOG_SAMPLES", "0") == "1" and len(train_dataset) > 0:
        sample_prompt = train_dataset[0]["prompt"]
        print("=" * 80)
        print("[Sample Prompt]")
        print(sample_prompt)
        print("[Mapped Target]")
        sample_hist = prompt2history[sample_prompt]
        print(history2target[sample_hist])
        print("=" * 80)

    trainer.train()
    trainer.save_model(args.output_dir)

    final_dir = os.path.join(args.output_dir, "final_checkpoint")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(args.output_dir, "rl_run_info.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reward_type": args.reward_type,
                "train_size": len(train_dataset),
                "eval_size": 0 if eval_dataset is None else len(eval_dataset),
                "sid2emb_size": len(sid2emb),
                "info_file": info_file,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Done] RL model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
