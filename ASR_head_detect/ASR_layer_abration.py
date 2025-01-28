import os
import copy
import pandas as pd
import torch
import shutil
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from meta_llama import MetaLlama3

def load_reject_phrases(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} が見つかりません．空リストを返します．")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    return phrases

def remove_layers_from_llama(base_model: AutoModelForCausalLM, remove_layer_indices: list):
    model_copy = copy.deepcopy(base_model)
    original_layers = model_copy.model.layers

    remain_layers = torch.nn.ModuleList(
        layer for i, layer in enumerate(original_layers)
        if i not in remove_layer_indices
    )
    
    model_copy.model.layers = remain_layers
    model_copy.config.num_hidden_layers = len(remain_layers)
    return model_copy

def compute_asr_with_metal(metal, prompts, reject_phrases, batch_size=4, generation_config=None):
    outputs = metal.predict_batch(prompts, generation_config=generation_config, batch_size=batch_size)
    reject_count = 0
    for text in outputs:
        if any(phrase in text for phrase in reject_phrases):
            reject_count += 1
    total = len(prompts)
    asr = reject_count / total if total > 0 else 0.0
    return asr

def ASR_layer_abration(dataset_path, reject_sentence_path, base_model_path):
    df = pd.read_csv(dataset_path)
    prompts = df["input"].tolist()
    reject_phrases = load_reject_phrases(reject_sentence_path)
    print("Reject phrases:", reject_phrases)
    print("[INFO] Loading base LlamaForCausalLM...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path) 
    _ = AutoTokenizer.from_pretrained(base_model_path)

    total_layers = len(base_model.model.layers)
    remove_indices_list = [[layer_idx] for layer_idx in range(total_layers)]
    generate_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.9,
        "do_sample": True
    }

    results = {}

    for remove_list in remove_indices_list:
        print("\n===============================")
        print(f"[START] Remove layer indices = {remove_list}")
        modified_model = remove_layers_from_llama(base_model, remove_list)
        temp_dir = f"./tmp_model_removed_{remove_list}"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        print(f"[SAVE] Model{temp_dir}")
        modified_model.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(temp_dir)

        print(f"[SAVED] Model{temp_dir}")
        metal = MetaLlama3(model_path=temp_dir)
        print(f"[COMPUTE] Compute ASR")
        asr = compute_asr_with_metal(
            metal=metal,
            prompts=prompts,
            reject_phrases=reject_phrases,
            batch_size=4,
            generation_config=generate_kwargs
        )
        results[str(remove_list)] = asr
        print(f"[RESULT] Remove {remove_list} => ASR = {asr:.4f}")
        del modified_model
        del metal
        torch.cuda.empty_cache()

        shutil.rmtree(temp_dir,ignore_errors=True)
        print(f"[CLEANUP] Removed temp_dir: {temp_dir}")
    x = list(results.keys())
    y = list(results.values())

    model_name = base_model_path.split('/')[-1]
    lang = reject_sentence_path.split('_')[-1]

    plt.figure(figsize=(8, 4))
    plt.bar(x, y, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Removed Layer Indices")
    plt.ylabel("ASR (Attack Success Rate)")
    plt.title(f"ASR by Removed Layer Indices {model_name} {lang}")
    plt.tight_layout()
    plt.savefig(f"asr_by_layer_removal_{model_name}_{lang}.png")
    plt.show()

    print("\n=== Summary of Results ===")
    for k, v in results.items():
        print(f"Removed {k} => ASR: {v:.4f}")


def main():
    ASR_layer_abration(
        dataset_path="../dataset_for_sahara/Multilingual_ja_600.csv",
        reject_sentence_path="./reject_keywords_ja.txt",
        base_model_path="../models/Llama-3.1-8B-Instruct"
    )
    ASR_layer_abration(
        dataset_path="../dataset_for_sahara/Multilingual_en_600.csv",
        reject_sentence_path="./reject_keywords_en.txt",
        base_model_path="../models/Llama-3.2-3B-Instruct"
    )
    ASR_layer_abration(
        dataset_path="../dataset_for_sahara/Multilingual_ja_600.csv",
        reject_sentence_path="./reject_keywords_ja.txt",
        base_model_path="../models/Llama-3.1-8B-Instruct"
    )
    ASR_layer_abration(
        dataset_path="../dataset_for_sahara/Multilingual_en_600.csv",
        reject_sentence_path="./reject_keywords_en.txt",
        base_model_path="../models/Llama-3.2-3B-Instruct"
    )

if __name__ == "__main__":
    main()
