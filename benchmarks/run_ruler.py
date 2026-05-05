import os
import sys
import json
import time
import glob
import shutil
import subprocess
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from main import apply_page_evict, reset_all_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--benchmark', type=str, default='synthetic')
    # Default to a small subset for quick benchmarking, can be expanded to all RULER tasks
    parser.add_argument('--tasks', type=str, nargs='+', default=["niah_single_1", "qa_1"]) 
    parser.add_argument('--method', type=str, choices=['page_evict', 'vanilla'], default='page_evict')
    args = parser.parse_args()

    ruler_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'RULER'))
    ruler_scripts = os.path.join(ruler_dir, 'scripts')
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'ruler', args.method)

    # Per-task tokens_to_generate, taken from RULER/scripts/data/synthetic/constants.py.
    # Hardcoding 128 over-generates for qa/vt and pollutes scoring; this map matches
    # what RULER itself passes to prepare.py so generations are cut where the metric
    # expects them to end.
    task_family_tokens = {
        "niah": 128,
        "variable_tracking": 30,
        "common_words_extraction": 120,
        "freq_words_extraction": 50,
        "qa": 32,
    }
    task_to_family = {
        "niah_single_1": "niah", "niah_single_2": "niah", "niah_single_3": "niah",
        "niah_multikey_1": "niah", "niah_multikey_2": "niah", "niah_multikey_3": "niah",
        "niah_multivalue": "niah", "niah_multiquery": "niah",
        "vt": "variable_tracking",
        "cwe": "common_words_extraction",
        "fwe": "freq_words_extraction",
        "qa_1": "qa", "qa_2": "qa",
    }
    
    # We must ensure that subprocess calls to 'python' use our virtual environment
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ruler_scripts}:{env.get('PYTHONPATH', '')}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Metrics tracking
    task_metrics = {}

    for task in args.tasks:
        print(f"\n{'='*50}\nEvaluating RULER Task: {task} at {args.max_length} length\n{'='*50}")
        
        # Scope data/pred dirs by max_length: prepare.py reuses an existing
        # validation.jsonl whenever sample count matches, so without this scoping a
        # second run at a different context length would silently reuse stale data.
        task_data_dir = os.path.join(results_dir, "data", str(args.max_length))
        task_pred_dir = os.path.join(results_dir, "pred", str(args.max_length))
        os.makedirs(task_data_dir, exist_ok=True)
        os.makedirs(task_pred_dir, exist_ok=True)

        # 1. Generate Synthetic Data using RULER's official scripts
        print("Preparing data...")
        cmd_prepare = [
            sys.executable, os.path.join(ruler_scripts, "data", "prepare.py"),
            "--save_dir", task_data_dir,
            "--benchmark", args.benchmark,
            "--task", task,
            "--tokenizer_path", args.model_id,
            "--tokenizer_type", "hf",
            "--max_seq_length", str(args.max_length),
            "--model_template_type", "meta-llama3",
            "--num_samples", "10" # Reduced samples for faster benchmark runs
        ]
        subprocess.run(cmd_prepare, env=env, check=True)

        # 2. Run Inference natively with our page-evict model
        print("Running inference...")
        validation_file = os.path.join(task_data_dir, task, "validation.jsonl")
        pred_file = os.path.join(task_pred_dir, f"{task}.jsonl")

        if not os.path.exists(validation_file):
            print(f"  Skipping {task}: prepare.py did not produce {validation_file}")
            continue

        if os.path.exists(pred_file):
            os.remove(pred_file)

        with open(validation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Use the per-task generation budget RULER's prepare.py used; fall back
        # to 128 only for tasks not in the map.
        tokens_to_generate = task_family_tokens.get(task_to_family.get(task, ""), 128)

        total_latency = 0.0
        total_tokens = 0
        torch.cuda.reset_peak_memory_stats()

        with open(pred_file, 'a', encoding='utf-8') as f_out:
            for line in tqdm(lines):
                sample = json.loads(line)
                prompt = sample['input']

                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[-1]
                
                if args.method == 'page_evict':
                    reset_all_states(layer_states, cache)

                start_t = time.time()
                with torch.no_grad():
                    generate_kwargs = dict(
                        **inputs,
                        max_new_tokens=tokens_to_generate,
                        use_cache=True,  # required for both methods so HF slices input_ids on decode
                        do_sample=False,
                    )
                    if args.method == 'page_evict':
                        generate_kwargs["past_key_values"] = cache
                    output_ids = model.generate(**generate_kwargs)
                latency = time.time() - start_t
                
                pred_ids = output_ids[0][input_len:]
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                
                total_latency += latency
                total_tokens += len(pred_ids)
                
                # Format output to match RULER's expected call_api.py prediction format.
                # `others` is required: evaluate.py reads line['others'].get('id', line['index'])
                # and KeyErrors if the field is missing.
                out_sample = {
                    'index': sample['index'],
                    'pred': pred_text,
                    'input': prompt,
                    'outputs': sample['outputs'],
                    'others': sample.get('others', {}),
                    'truncation': sample.get('truncation', 0),
                    'length': sample.get('length', -1),
                }
                f_out.write(json.dumps(out_sample) + '\n')

        # Record metrics for the task
        task_metrics[task] = {
            "throughput_tokens_per_sec": total_tokens / total_latency if total_latency > 0 else 0,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / (1024**3)
        }

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(task_metrics, f, indent=4)

    # 3. Evaluate Predictions using RULER's official scripts
    print("Evaluating predictions...")
    cmd_eval = [
        sys.executable, os.path.join(ruler_scripts, "eval", "evaluate.py"),
        "--data_dir", task_pred_dir,
        "--benchmark", args.benchmark
    ]
    subprocess.run(cmd_eval, env=env, check=True)
    
    # RULER's evaluate.py writes 'summary.csv' for multi-task runs and
    # 'summary-{task}.csv' for single-task runs. Copy whichever it produced into the
    # method-level results dir so plot_results.py can find it without guessing paths.
    for src in glob.glob(os.path.join(task_pred_dir, "summary*.csv")):
        shutil.copy2(src, os.path.join(results_dir, os.path.basename(src)))

if __name__ == '__main__':
    main()
