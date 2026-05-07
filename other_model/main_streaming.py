# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import sys
import os
from pipeline_streaming import StreamingPipeline
#from tokenizer.tokenizer import VanillaTokenizer
from tokenizer.streaming_tokenizer import StreamingTokenizer
from model import SASRec, TIGER, ActionPiece, DiffGRM, LLaDA_Rec


def parse_args():

    parser = argparse.ArgumentParser(description='Streaming Training for Recommendation Models')
    
    parser.add_argument('--model', type=str, default='TIGER', 
                        help='Model name (default: TIGER)')
    parser.add_argument('--dataset', type=str, default='StreamingData', 
                        help='Dataset name (default: StreamingData)')
    parser.add_argument('--category', type=str, default=None, 
                        help='Dataset category (optional)')
    

    parser.add_argument('--json_config_path', type=str, required=True,
                        help='Path to JSON config file containing interaction and item files')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint for resuming training')
    

    parser.add_argument('--task', type=str, default='is_click',
                        help='Interaction task type (default: is_click)')
    parser.add_argument('--item_history_days', type=int, default=7,
                        help='Number of days for item history (default: 7)')
    parser.add_argument('--epochs_per_round', type=int, default=1,
                        help='Number of epochs per training round (default: 5)')
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help='Maximum sequence length (default: 50)')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='Minimum sequence length for users (default: 5)')
    

    parser.add_argument('--training_mode', type=str, default='continual',
                        choices=['continual', 'from_scratch', 'from_checkpoint'],
                        help='Training mode (default: continual)')
 

    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--model_size', type=str, default='tiny',
                        help='Model size: tiny, small, base, large (default: small)')
    parser.add_argument('--num_beams', type=int, default=50,
                        help='Number of beams for beam search (default: 50)')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='Training batch size (default: 128)')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size (default: 32)')
    
 
    parser.add_argument('--output_dir', type=str, default='./outputs/streaming',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_every_round', action='store_true',
                        help='Save checkpoint after every round')
    

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'eval_all', 'resume', 'plan','eval_prev_ckpt_last_day'],
                        help='Running mode (default: train)')
    parser.add_argument('--resume_from', type=int, default=1,
                        help='Round number to resume training from (default: 1)')
    parser.add_argument('--data_mode',type=str,default='streaming',choices=['streaming', 'full_user'],
                        help='Data mode: streaming or full_user')

    parser.add_argument('--max_history_len', type=int, default=999,
                        help='Maximum history length per user (default: 100)')

    parser.add_argument('--max_train_samples_per_user', type=int, default=1000,
                        help='Maximum training samples per user (default: 50)')
    parser.add_argument('--no_accumulate', action='store_true',
                        help='Disable history accumulation (each round independent)')

    parser.add_argument('--last_day_target_limit', type=int, default=50,
                    help='For eval_prev_ckpt_last_day mode: number of first interactions on the last day used as targets per user')
    parser.add_argument('--eval_round_num', type=int, default=None,
                    help='Optional round number for eval modes. For eval_prev_ckpt_last_day, default is the last round')

    
    # wandb 
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='tiger-streaming', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='Wandb tags')

    return parser.parse_known_args()


def parse_extra_args(unparsed: list) -> dict:

    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            continue
        key, value = text_arg.split('=', 1)
        key = key.lstrip('-')
        
        try:
            import json
            value = json.loads(value)
        except:
            try:
                import ast
                value = ast.literal_eval(value)
            except:
                pass
        
        args[key] = value
    return args


def main():

    args, unparsed_args = parse_args()
    extra_config = parse_extra_args(unparsed_args)
    
    accumulate_history = not args.no_accumulate

    config_dict = {
        'lr': args.lr,
        'model_size': args.model_size,
        'num_beams': args.num_beams,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'output_dir': args.output_dir,
        'save_every_round': args.save_every_round,
    }
    config_dict.update(extra_config)
    

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Streaming Training Pipeline")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"JSON Config: {args.json_config_path}")
    print(f"Task: {args.task}")
    print(f"Item History Days: {args.item_history_days}")
    print(f"Epochs per Round: {args.epochs_per_round}")
    print(f"Training Mode: {args.training_mode}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
  
    pipeline = StreamingPipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        category=args.category,
        json_config_path=args.json_config_path,
        checkpoint_path=args.checkpoint,
        tokenizer=StreamingTokenizer,
        config_dict=config_dict,
        task=args.task,
        epochs_per_round=args.epochs_per_round,
        item_history_days=args.item_history_days,
        max_seq_len=args.max_seq_len,
        min_seq_len=args.min_seq_len,
        training_mode=args.training_mode,
        data_mode=args.data_mode,
        max_history_len=args.max_history_len,
        accumulate_history=accumulate_history,
        max_train_samples_per_user=args.max_train_samples_per_user,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
    

    if args.mode == 'train':
        results = pipeline.run()
    elif args.mode == 'eval':
        results = pipeline.evaluate_only()
    elif args.mode == 'eval_all':
        results = pipeline.evaluate_all_rounds()
    elif args.mode == 'resume':
        results = pipeline.resume_training(from_round=args.resume_from)
    elif args.mode == 'plan':
        pipeline.print_training_plan()
        results = {'status': 'plan_printed'}
    elif args.mode == 'eval_prev_ckpt_last_day':
        results = pipeline.evaluate_last_day_with_prev_ckpt(
            eval_round=args.eval_round_num,
            target_limit=args.last_day_target_limit,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print(f"\n{'='*60}")
    print(f"Execution Complete")
    print(f"{'='*60}")
    
    if args.mode in ['train', 'resume']:
        print(f"Total Rounds: {results.get('total_rounds', 'N/A')}")
        print(f"Best Round: {results.get('best_round', 'N/A')}")
        if results.get('best_val_score') and results.get('best_val_score') != float('-inf'):
            print(f"Best Val Score: {results.get('best_val_score'):.4f}")
    elif args.mode == 'eval':
        print(f"Test Results: {results.get('test_results', 'N/A')}")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    main()
