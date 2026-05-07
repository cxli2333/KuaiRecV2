import pandas as pd
import json
import os
import ast
from tqdm import tqdm


# '.../streaming'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR,'..', 'streaming_data', 'info')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR,'..', 'streaming_data','test','test_data' ,'info')

#DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR, 'Full_Month_file_paths.json')
DEFAULT_JSON_PATH = os.path.join(SCRIPT_DIR,'..','streaming_data','test','test_file_paths.json')

def parse_sid_list(sid_str):
     
    if not isinstance(sid_str, str) or not sid_str:
        return None, None
    
    raw_tokens = sid_str.split(',')

    prefixes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    token_list = []

    token_list.append("<sid_start_flag>")
    
    for i, t in enumerate(raw_tokens):
        t = t.strip()
        if not t: continue
        
        if i < len(prefixes):
            prefix = prefixes[i]
        else:
            prefix = f"l{i}" # fallback for very deep hierarchies
            
        new_token = f"<{prefix}_{t}>"
        token_list.append(new_token)

    token_list.append("<sid_end_flag>")
    
    concatenated_sid = "".join(token_list)
    
    return token_list, concatenated_sid

def create_master_files(file_paths_json, output_dir=DEFAULT_OUTPUT_DIR):
    
    try:
        with open(file_paths_json, 'r') as f:
            paths = json.load(f)
        item_files = paths['item_files']
    except Exception as e:
        print(f"Error: Unable to read {file_paths_json}。 Please ensure that it contains an 'item_files' list.")
        print(e)
        return

    os.makedirs(output_dir, exist_ok=True)
    
    master_item_df = pd.DataFrame()
    print("Loading all item files...")
    
    column_names = ['sid_list', 'emb_vec', 'item_id', 'tag_list']
    
    for item_file in tqdm(item_files):
        try:
            
            df = pd.read_csv(
                item_file, 
                sep='\t', 
                header=None,
                names=column_names,
                dtype={'item_id': str} 
            )
            master_item_df = pd.concat([master_item_df, df])
        except Exception as e:
            print(f"Warning: Failed to load {item_file}")
            print(f"Error details: {e}")

    
    master_item_df = master_item_df.drop_duplicates(subset=['item_id'], keep='last')
    print(f"A total of {len(master_item_df)} unique items were found")

    
    item_json_path = os.path.join(output_dir, 'master.item.json')
    index_json_path = os.path.join(output_dir, 'master.index.json')
    info_txt_path = os.path.join(output_dir, 'master.info.txt')

    item_metadata = {}    # for master.item.json
    item_to_semantic = {} # for master.index.json
    info_txt_lines = []   # for master.info.txt

    print("Processing the unique item and generating the main file (untitled mode)...")
    for _, row in tqdm(master_item_df.iterrows(), total=len(master_item_df)):
        try:
            original_item_id = str(row['item_id']).replace('.0', '')
            
            sid_token_list, concatenated_sid = parse_sid_list(row['sid_list'])
            if sid_token_list is None:
                continue
            item_to_semantic[original_item_id] = sid_token_list 

            synthetic_title = f"Item {original_item_id}" 


            item_metadata[original_item_id] = {
                "title": synthetic_title 
            }
            
            info_txt_lines.append(f"{concatenated_sid}\t{synthetic_title}\t{original_item_id}\n")

        except Exception as e:
            
            pass

    with open(item_json_path, 'w') as f:
        json.dump(item_metadata, f, indent=2)
    print(f"Metadata saved to: {item_json_path}")

    with open(index_json_path, 'w') as f:
        json.dump(item_to_semantic, f, indent=2)
    print(f"Saved SID index to: {index_json_path}")

    with open(info_txt_path, 'w') as f:
        f.writelines(info_txt_lines)
    print(f"Saved Info Txt to: {info_txt_path}")

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description="Create the main index file")
    parser.add_argument(
        '--file_paths_json', 
        type=str, 
        default=DEFAULT_JSON_PATH, 
        help="A JSON file containing a list of file_paths"
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR, 
        help="master file output path"
    )
    args = parser.parse_args()
    
    create_master_files(args.file_paths_json,args.output_path)