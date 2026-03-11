import argparse
import json
import re
import sqlite3
import threading
import random
import logging
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Any, Union, List, Dict

from utils.prepare_input_seq import get_input_seq


def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl2sql_ckpt_path", type = str)
    parser.add_argument("--dataset_name", type = str, help = "the name of the dataset")
    parser.add_argument("--input_file", type = str, help = "the input file path (prompts)")
    parser.add_argument("--output_file", type = str, help = "the output file path (results)")
    parser.add_argument("--database_path", type = str, help = "the path of the database")
    parser.add_argument("--tensor_parallel_size", type = int, help = "the number of used GPUs", default = 4)
    parser.add_argument("--n", type = int, help = "the number of generated responses", default = 4)
    parser.add_argument("--temperature", type = float, help = "temperature of llm's sampling", default = 1.0)
    parser.add_argument("--output_format", type = str, help = "the format of the output file", default = "json")
    parser.add_argument("--table_value_cache_path", type = str)
    parser.add_argument("--table_info_cache_path", type = str)
    parser.add_argument("--think_mode", type = bool, default=False)
    parser.add_argument("--input_format", type = str, default="json", choices=["json", "parquet"])

    opt = parser.parse_args()
    print(opt)

    tokenizer = AutoTokenizer.from_pretrained(opt.nl2sql_ckpt_path, trust_remote_code=True, local_files_only=True)
    
    if "Qwen2.5-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [151645] # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
    elif "deepseek-coder-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in opt.nl2sql_ckpt_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [128009, 128001]
    elif "granite-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [0] # <|end_of_text|> is the end token of granite-3.1 and granite-code
    elif "starcoder2-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [0] # <|end_of_text|> is the end token of starcoder2
    elif "Codestral-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [2]
    elif "Mixtral-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [2]
    elif "OmniSQL-" in opt.nl2sql_ckpt_path:
        stop_token_ids = [151645] # OmniSQL uses the same tokenizer as Qwen2.5
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]

    print("stop_token_ids:", stop_token_ids)
    
    max_model_len = 8192 # used to allocate KV cache memory in advance
    max_input_len = 6144
    max_output_len = 2048 # (max_input_len + max_output_len) must <= max_model_len
    
    print("max_model_len:", max_model_len)
    print("temperature:", opt.temperature)

    if "Qwen3-" in opt.nl2sql_ckpt_path:
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            top_p=0.8,
            top_k=20,
        )
    else:
        sampling_params = SamplingParams(
            temperature = opt.temperature, 
            max_tokens = max_output_len,
            n = opt.n,
            stop_token_ids = stop_token_ids
        )        

    llm = LLM(
        model = opt.nl2sql_ckpt_path,
        dtype = "bfloat16", 
        tensor_parallel_size = opt.tensor_parallel_size,
        max_model_len = max_model_len,
        gpu_memory_utilization = 0.92,
        swap_space = 42,
        enforce_eager = True,
        disable_custom_all_reduce = True,
        trust_remote_code = True
    )
    
    if opt.input_format == "parquet":
        import pandas as pd
        df = pd.read_parquet(opt.input_file)
        input_dataset = df.to_dict('records')
        chat_prompts = []
        for row in input_dataset:
            prompt = row['prompt']
            if hasattr(prompt, 'tolist'):
                prompt = prompt.tolist()  # numpy array → list of dicts
            chat_prompts.append(tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            ))
        # Extract gold SQL into each row for output
        for row in input_dataset:
            rm = row.get('reward_model')
            if isinstance(rm, dict):
                row['gold_sql'] = rm.get('ground_truth', {}).get('sql', '')
    else:
        input_dataset = json.load(open(opt.input_file))
        if "Qwen3-" in opt.nl2sql_ckpt_path:
            chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": get_input_seq(
                    data, opt.database_path, opt.dataset_name, opt.table_value_cache_path, opt.table_info_cache_path
                    )}],
                add_generation_prompt = True, tokenize = False
            ) for data in input_dataset]
        else:
            chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": get_input_seq(
                    data, opt.database_path, opt.dataset_name, opt.table_value_cache_path, opt.table_info_cache_path
                    )}],
                add_generation_prompt = True, tokenize = False, enable_thinking=opt.think_mode
            ) for data in input_dataset]

    outputs = llm.generate(chat_prompts, sampling_params)
    
    results = []
    if opt.output_format == "json":
        for data, output in zip(input_dataset, outputs):
            responses = [o.text for o in output.outputs]
            sqls  = [parse_response(response) for response in responses]
            
            data["responses"] = responses
            # for i in range(len(sqls)):
            #     sqls[i] = sqls[i].replace('\n', ' ').strip() if len(sqls[i]) > 0 else 'SELECT'
            data["pred_sqls"] = sqls
            results.append(data)

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                return super().default(obj)

        with open(opt.output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(results, indent = 2, ensure_ascii = False, cls=_NumpyEncoder))
            
    elif opt.output_format == "txt":
        for data, output in zip(input_dataset, outputs):
            responses = [o.text for o in output.outputs]
            sqls  = [parse_response(response) for response in responses]
            
            if opt.n == 1:
                results.append(sqls[0].replace('\n', ' ').strip() if len(sqls[0]) > 0 else 'SELECT')
            else:
                for i in range(len(sqls)):
                    sqls[i] = sqls[i].replace('\n', ' ').strip() if len(sqls[i]) > 0 else 'SELECT'
                results.append(sqls)
        with open(opt.output_file, "w", encoding = "utf-8") as f:
            f.write("\n".join(results))
    else:
        raise ValueError(f"Invalid output format: {opt.output_format}")