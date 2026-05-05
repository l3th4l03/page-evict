import os
import json
import time
import torch
import argparse
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from main import apply_page_evict, reset_all_states

# We use the prompt building from LongBench pred.py
def build_chat(tokenizer, prompt):
    # For Llama-3.1-Instruct
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--max_length', type=int, default=8192) # Max context size
    parser.add_argument('--method', type=str, choices=['page_evict', 'vanilla'], default='page_evict')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer
    print(f"Loading tokenizer {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    print(f"Loading model {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    
    layer_states = None
    cache = None
    if args.method == 'page_evict':
        print("Applying page-evict monkeypatch...")
        layer_states, cache = apply_page_evict(model)
    else:
        print("Running vanilla baseline (no monkeypatch)...")
    
    model.eval()

    # Load LongBench configs
    longbench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'LongBench'))
    eval_dir = os.path.join(longbench_dir, "LongBench")
    
    with open(os.path.join(eval_dir, "config", "dataset2prompt.json"), "r") as f:
        dataset2prompt = json.load(f)
    with open(os.path.join(eval_dir, "config", "dataset2maxlen.json"), "r") as f:
        dataset2maxlen = json.load(f)

    # Canonical name → HF config name. The local HF cache holds LongBench-E ("_e")
    # variants for most tasks plus the standard variant for the three that have no
    # length-balanced version. The canonical (base) name is what `dataset2prompt`,
    # `dataset2maxlen`, and `eval.py`'s `dataset2metric` are keyed on, and is also
    # what `eval.py` uses to pick the metric from the prediction filename.
    dataset_to_hf_config = {
        "narrativeqa": "narrativeqa",
        "qasper": "qasper_e",
        "multifieldqa_en": "multifieldqa_en_e",
        "hotpotqa": "hotpotqa_e",
        "2wikimqa": "2wikimqa_e",
        "musique": "musique",
        "gov_report": "gov_report_e",
        "qmsum": "qmsum",
        "multi_news": "multi_news_e",
        "trec": "trec_e",
        "triviaqa": "triviaqa_e",
        "samsum": "samsum_e",
        "passage_count": "passage_count_e",
        "passage_retrieval_en": "passage_retrieval_en_e",
        "lcc": "lcc_e",
        "repobench-p": "repobench-p_e",
    }
    datasets = list(dataset_to_hf_config.keys())

    # To safely use LongBench's eval.py without modifying it, we save predictions
    # directly into LongBench/LongBench/pred/{method}/
    eval_dir = os.path.join(longbench_dir, "LongBench")
    out_dir = os.path.join(eval_dir, "pred", args.method)
    os.makedirs(out_dir, exist_ok=True)
    
    # We will also keep a copy in our results directory
    final_out_dir = os.path.join(os.path.dirname(__file__), "results", "longbench", args.method)
    os.makedirs(final_out_dir, exist_ok=True)
    
    # Metrics tracking
    task_metrics = {}

    for dataset in datasets:
        hf_config = dataset_to_hf_config[dataset]
        print(f"\nEvaluating dataset: {dataset} (HF config: {hf_config})")
        try:
            data = load_dataset('THUDM/LongBench', hf_config, split='test')
        except (ValueError, FileNotFoundError) as e:
            print(f"  Skipping {dataset}: could not load HF config '{hf_config}': {e}")
            continue
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        out_path = os.path.join(out_dir, f"{dataset}.jsonl")
        
        # If output file already exists, we skip or could resume. For simplicity, write new.
        if os.path.exists(out_path):
            os.remove(out_path)

        total_latency = 0.0
        total_tokens = 0
        torch.cuda.reset_peak_memory_stats()

        for json_obj in tqdm(data):
            prompt = prompt_format.format(**json_obj)
            
            # Truncate prompt if it exceeds max_length
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized_prompt) > args.max_length - max_gen:
                half = int((args.max_length - max_gen) / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
            # Build chat prompt (unless it's a dataset where chat template hurts)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt)
            
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            context_length = input_ids.shape[-1]
            
            if args.method == 'page_evict':
                reset_all_states(layer_states, cache)

            start_t = time.time()
            with torch.no_grad():
                generate_kwargs = dict(
                    input_ids=input_ids,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=True,  # required for both methods so HF slices input_ids on decode
                )
                if args.method == 'page_evict':
                    # Pass the shim cache so HF's seq_length tracking matches our buffer.
                    generate_kwargs["past_key_values"] = cache
                output = model.generate(**generate_kwargs)[0]
            latency = time.time() - start_t
                
            pred_ids = output[context_length:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            total_latency += latency
            total_tokens += len(pred_ids)
            
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')

        # Record metrics for the task
        task_metrics[dataset] = {
            "throughput_tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3)
        }
        
    # Save metrics
    with open(os.path.join(final_out_dir, "metrics.json"), "w") as f:
        json.dump(task_metrics, f, indent=4)
        
    # Run Evaluation
    print("Running LongBench Evaluation...")
    cmd_eval = [sys.executable, "eval.py", "--model", args.method]
    subprocess.run(cmd_eval, cwd=eval_dir, check=True)
    
    # Copy results to final directory
    import shutil
    shutil.copy2(os.path.join(out_dir, "result.json"), os.path.join(final_out_dir, "result.json"))

if __name__ == '__main__':
    main()
