import pandas as pd
import json
import os
import ast
from tqdm import tqdm
import random

def load_master_lookups(master_item_json_path, master_index_json_path):
    print("Loading master.item.json ...")
    try:
        with open(master_index_json_path, 'r') as f:
            item_to_semantic = json.load(f)
        with open(master_item_json_path, 'r') as f:
            item_metadata = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load the main JSON file: {e}")
        return None, None, None

    item_id_to_sid_str = {}
    item_id_to_title = {} 
    valid_item_ids = set()

    for item_id, tokens in item_to_semantic.items():
        if item_id in item_metadata:
            item_id_to_sid_str[item_id] = "".join(tokens)
            item_id_to_title[item_id] = item_metadata[item_id].get('title', f"Item {item_id}")
            valid_item_ids.add(item_id)
            
    print(f"Lookup table loaded successfully (total of {len(valid_item_ids)} items)。")
    return item_id_to_sid_str, item_id_to_title, valid_item_ids


def prepare_data_for_day(
    interaction_file, 
    output_dir, 
    item_sid_map, 
    item_title_map, 
    valid_items,
    max_history_len=50,
    valid_behaviors=None
):
    if valid_behaviors is None or len(valid_behaviors) == 0:
        valid_behaviors = ['is_click']
    
    print(f"[Data Preparation] Currently filtering the following behaviors:{valid_behaviors}")
    
    try:
        header_df = pd.read_csv(interaction_file, sep='\t', nrows=1)
        existing_cols = header_df.columns.tolist()
    except Exception as e:
        print(f"Error: Unable to peek file header: {e}")
        return None, None
        
    if 'user_str' in existing_cols:
        user_col = 'user_str'
    elif 'user' in existing_cols:
        user_col = 'user'
    else:
        print(f"Error: Neither 'user' nor 'user_str' found in {interaction_file}. Columns are: {existing_cols}")
        return None, None

    cols_to_read = [user_col, 'item', 'time_ms'] + valid_behaviors

    df = pd.read_csv(
        interaction_file, 
        sep='\t', 
        usecols=cols_to_read, 
        dtype={user_col: str, 'item': str}
    )
    df.rename(columns={user_col: 'user'}, inplace=True)

    original_count = len(df)

    mask = pd.Series([False] * len(df), index=df.index)
    for col in valid_behaviors:
        if col in df.columns:
            mask = mask | (df[col] == 1)

    df = df[mask]
    filtered_count = len(df)
    print(f"[Filtering] Original: {original_count} rows -> After filtering: {filtered_count} rows")

    if df.empty:
        print(" [Warning] No remaining data after filtering")
        return None, None

    df['item'] = df['item'].astype(str).str.replace(r'\.0$', '', regex=True)
    df = df[df['item'].isin(valid_items)]
    df = df.sort_values(by=['user', 'time_ms'])
    
    test_sequences_csv = []
    train_sequences_jsonl = []
    
    grouped = df.groupby('user')
    
    for user_id, group in tqdm(grouped, desc="Processing users"):
        interactions = []
        for item_id_str in group['item']:
            sid = item_sid_map.get(item_id_str)
            title = item_title_map.get(item_id_str, f"Item {item_id_str}")
            if sid:
                interactions.append((sid, title)) 
        
        if len(interactions) < 2:
            continue

        test_item = interactions[-1]
        train_items = interactions[:-1]
        
        test_sid, test_title = test_item
        train_sids = [item[0] for item in train_items]
        train_titles = [item[1] for item in train_items]
        
        hist_sids = train_sids[-max_history_len:]
        hist_titles = train_titles[-max_history_len:]
        test_sequences_csv.append([
            user_id, 
            str(hist_sids), 
            test_sid,
            str(hist_titles), 
            test_title        
        ])
        
        if train_sids:
            train_sequences_jsonl.append({
                "user_id": user_id,
                "sequence_sids": train_sids,   
                "sequence_titles": train_titles 
            })
            
    test_columns = ['user_id', 'history_item_sid', 'item_sid', 'history_item_title', 'item_title'] 
    test_path = os.path.join(output_dir, 'test_T.csv')
    pd.DataFrame(test_sequences_csv, columns=test_columns).to_csv(test_path, index=False)
    
    train_path = os.path.join(output_dir, 'train_sequences.jsonl')
    with open(train_path, 'w') as f:
        for line_data in train_sequences_jsonl:
            f.write(json.dumps(line_data) + '\n')
    
    print(f"  (New) Training sequence file (JSONL): {len(train_sequences_jsonl)} lines")
    print(f"  (New) Test sequence file (csv): {len(test_sequences_csv)} lines")
    
    return train_path, test_path


def prepare_rl_data_with_sampling(
    interaction_file,
    master_item_path,
    master_index_path,
    output_dir,
    train_day,
    sample_rate=0.1,
    max_history_len=20,
    valid_behaviors=['is_click'],
):
    print(f"\n--- Preparing RL data for train_day={train_day} ---")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load master lookup
    item_id_to_sid_str, item_id_to_title, valid_item_ids = load_master_lookups(
        master_item_path, master_index_path
    )

    # 2) Read interaction file
    df = pd.read_csv(interaction_file, sep='\t')
    existing_cols = set(df.columns)

    if 'user_str' in existing_cols:
        user_col = 'user_str'
    elif 'user' in existing_cols:
        user_col = 'user'
    else:
        raise ValueError(f"Cannot find user column in {interaction_file}. Existing columns: {list(df.columns)}")

    if 'time_ms' not in existing_cols:
        raise ValueError(f"Cannot find 'time_ms' in {interaction_file}. Existing columns: {list(df.columns)}")

    if 'item_id' in df.columns:
        item_col = 'item_id'
    elif 'item' in df.columns:
        item_col = 'item'
    else:
        raise ValueError(f"Cannot find item column ('item_id' or 'item') in {interaction_file}. Existing columns: {list(df.columns)}")

    #  
    df.rename(columns={user_col: 'user', item_col: 'item_id'}, inplace=True)
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df.dropna(subset=['time_ms'])
    
    #  
    all_behavior_cols = [col for col in df.columns if col.startswith('is_')]
    print(f"[Info] Captured behavior columns from data: {all_behavior_cols}")

    missing_cols = [col for col in valid_behaviors if col not in existing_cols]
    if missing_cols:
        raise ValueError(f"Valid behaviors columns missing: {missing_cols}. Existing columns: {list(df.columns)}")
        
    #  
    mask = df[valid_behaviors].eq(1).all(axis=1)
    df = df[mask]

    # 4) Keep only valid items
    df["item_id"] = df["item_id"].astype("string").str.strip()
    df['item_id'] = df['item_id'].str.replace(r'\.0$', '', regex=True)
    df_matched = df[df['item_id'].isin(valid_item_ids)].copy()

    if len(df_matched) == 0:
        print("❌ No valid interactions matched with master item/index.")
        return None, None

    # 5) Sort by user and time
    df_matched = df_matched.sort_values(by=['user', 'time_ms'])
    grouped = df_matched.groupby('user')

    all_train_pairs = []
    test_sequences_csv = []

    #  
    for user_id, g in grouped:
        seq_items = []
        for _, row in g.iterrows():
            iid = str(row['item_id'])
            sid_str = item_id_to_sid_str.get(iid, None)
            title = item_id_to_title.get(iid, f"Item {iid}")
            
            if sid_str is not None:
                #  
                behaviors_dict = {col: int(row.get(col, 0)) for col in all_behavior_cols}
                seq_items.append((sid_str, title, behaviors_dict))

        if len(seq_items) < 2:
            continue

        # 
        test_target = seq_items[-1]
        test_hist = seq_items[:-1][-max_history_len:]
        
        test_sequences_csv.append([
            user_id,
            str([x[0] for x in test_hist]),  # history sids
            test_target[0],                  # target sid
            str([x[1] for x in test_hist]),  # history titles
            test_target[1]                   # target title
        ])

        #  
        full_train_seq = seq_items[:-1]
        for k in range(1, len(full_train_seq)):
            history_part = [x[0] for x in full_train_seq[:k]][-max_history_len:]
            target_part = full_train_seq[k][0]
            target_behaviors = full_train_seq[k][2]  #  
            
            all_train_pairs.append((user_id, str(history_part), target_part, target_behaviors))

 
    test_columns = ['user_id', 'history_item_sid', 'item_sid', 'history_item_title', 'item_title']
    test_path = os.path.join(output_dir, 'test_T.csv')
    pd.DataFrame(test_sequences_csv, columns=test_columns).to_csv(test_path, index=False)

    total_samples = len(all_train_pairs)
    print(f"Total RL train pairs before sampling: {total_samples}")

    if total_samples == 0:
        print("❌ Valid interactions exist, but sequence length is too short for RL training.")
        return None, test_path

    sample_rate = min(max(float(sample_rate), 0.0), 1.0)
    random.shuffle(all_train_pairs)

    keep_num = int(total_samples * sample_rate)
    if sample_rate > 0 and keep_num == 0:
        keep_num = 1
    keep_num = min(keep_num, total_samples)

    sampled_pairs = all_train_pairs[:keep_num]

     
    train_path = os.path.join(output_dir, 'train_rl_sampled.tsv')
    with open(train_path, 'w', encoding='utf-8') as f:
 
        header_cols = ["user_id", "history", "target"] + all_behavior_cols
        f.write("\t".join(header_cols) + "\n")
        
        for user_id, history_sid_list_str, target_sid, behaviors_dict in sampled_pairs:
             
            row_data = [str(user_id), history_sid_list_str, str(target_sid)]
            
            for col in all_behavior_cols:
                row_data.append(str(behaviors_dict[col]))
                
            f.write("\t".join(row_data) + "\n")

    print(f"✅ Saved RL train file: {train_path}, sampled {len(sampled_pairs)} / {total_samples}")
    print(f"✅ Saved RL test file : {test_path}")

    return train_path, test_path

def _read_filtered_interaction_df(
    interaction_file,
    valid_items,
    valid_behaviors=None
):
    if valid_behaviors is None or len(valid_behaviors) == 0:
        valid_behaviors = ["is_click"]

    try:
        header_df = pd.read_csv(interaction_file, sep="\t", nrows=1)
        existing_cols = header_df.columns.tolist()
    except Exception as e:
        print(f"Error: Unable to peek file header: {e}")
        return None

    if "user_str" in existing_cols:
        user_col = "user_str"
    elif "user" in existing_cols:
        user_col = "user"
    else:
        print(f"Error: Neither 'user' nor 'user_str' found in {interaction_file}")
        return None

    cols_to_read = [user_col, "item", "time_ms"] + valid_behaviors
    cols_to_read = [c for c in cols_to_read if c in existing_cols]

    df = pd.read_csv(
        interaction_file,
        sep="\t",
        usecols=cols_to_read,
        dtype={user_col: str, "item": str}
    )
    df.rename(columns={user_col: "user"}, inplace=True)

    mask = pd.Series([False] * len(df), index=df.index)
    for col in valid_behaviors:
        if col in df.columns:
            mask = mask | (df[col] == 1)

    df = df[mask].copy()
    if df.empty:
        return df

    df["item"] = df["item"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df[df["item"].isin(valid_items)].copy()
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df.dropna(subset=["time_ms"])
    df = df.sort_values(by=["user", "time_ms"]).reset_index(drop=True)
    return df


def prepare_list_eval_from_two_days(
    history_day_file,
    target_day_file,
    output_dir,
    item_sid_map,
    item_title_map,
    valid_items,
    max_history_len=50,
    target_len=50,
    valid_behaviors=None,
    strict_target_len=True
):
     
    if valid_behaviors is None or len(valid_behaviors) == 0:
        valid_behaviors = ["is_click"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[ListEval] history day file: {history_day_file}")
    print(f"[ListEval] target day file : {target_day_file}")
    print(f"[ListEval] valid behaviors : {valid_behaviors}")

    hist_df = _read_filtered_interaction_df(
        history_day_file,
        valid_items=valid_items,
        valid_behaviors=valid_behaviors
    )
    tgt_df = _read_filtered_interaction_df(
        target_day_file,
        valid_items=valid_items,
        valid_behaviors=valid_behaviors
    )

    if hist_df is None or hist_df.empty:
        print("[ListEval] history day has no valid data")
        return None

    if tgt_df is None or tgt_df.empty:
        print("[ListEval] target day has no valid data")
        return None

    hist_grouped = hist_df.groupby("user")
    tgt_grouped = tgt_df.groupby("user")

    common_users = sorted(set(hist_df["user"].unique()) & set(tgt_df["user"].unique()))
    print(f"[ListEval] common users: {len(common_users)}")

    rows = []

    for user_id in tqdm(common_users, desc="Building list-eval test set"):
        hist_group = hist_grouped.get_group(user_id)
        tgt_group = tgt_grouped.get_group(user_id)

        hist_interactions = []
        for item_id_str in hist_group["item"].tolist():
            sid = item_sid_map.get(item_id_str)
            title = item_title_map.get(item_id_str, f"Item {item_id_str}")
            if sid:
                hist_interactions.append((sid, title))

        tgt_interactions = []
        for item_id_str in tgt_group["item"].tolist():
            sid = item_sid_map.get(item_id_str)
            title = item_title_map.get(item_id_str, f"Item {item_id_str}")
            if sid:
                tgt_interactions.append((sid, title))

        if len(hist_interactions) == 0:
            continue
        if len(tgt_interactions) == 0:
            continue
        if strict_target_len and len(tgt_interactions) < target_len:
            continue

        hist_interactions = hist_interactions[-max_history_len:]
        tgt_interactions = tgt_interactions[:target_len]

        hist_sids = [x[0] for x in hist_interactions]
        hist_titles = [x[1] for x in hist_interactions]

        target_sids = [x[0] for x in tgt_interactions]
        target_titles = [x[1] for x in tgt_interactions]

        if len(target_sids) == 0:
            continue

        rows.append([
            user_id,
            json.dumps(hist_sids, ensure_ascii=False),
            target_sids[0],
            json.dumps(hist_titles, ensure_ascii=False),
            target_titles[0],
            json.dumps(target_sids, ensure_ascii=False),
            json.dumps(target_titles, ensure_ascii=False),
        ])

    columns = [
        "user_id",
        "history_item_sid",
        "item_sid",
        "history_item_title",
        "item_title",
        "output",
        "output_title_list",
    ]

    test_path = os.path.join(output_dir, "test_list_eval.csv")
    pd.DataFrame(rows, columns=columns).to_csv(test_path, index=False)

    print(f"[ListEval] saved to: {test_path}")
    print(f"[ListEval] samples : {len(rows)}")

    return test_path
