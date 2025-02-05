import os
import copy
import pandas as pd
import torch
import shutil
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from meta_llama import MetaLlama3

# ------ ここに MetaLlama3クラスの定義があると仮定 (既存コード) ------
# from meta_llama3 import MetaLlama3  # 必要であれば別ファイルからimport

def load_reject_phrases(file_path: str) -> List[str]:
    """
    攻撃（拒否）ワード/フレーズが改行区切りで格納されたテキストファイルを読み込み
    リストとして返す
    """
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} が見つかりません．空リストを返します．")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        phrases = [line.strip() for line in f if line.strip()]
    return phrases

def compute_asr_with_metal(model, prompts, reject_phrases, batch_size=4, generation_config=None):
    """
    MetaLlama3インスタンスを使ってバッチ推論し，ASRを計算する関数．
    生成結果に reject_phrases のいずれかが含まれたら "reject" としてカウント．
    Attach Success Rateを返す
    """
    outputs = model.predict_batch(prompts, generation_config=generation_config, batch_size=batch_size)
    reject_count = 0
    for text in outputs:
        # 拒否フレーズのどれかが含まれていたらreject
        if any(phrase in text for phrase in reject_phrases):
            reject_count += 1
    total = len(prompts)
    asr = ( 1-(reject_count / total)) if total> 0 else 0.0
    return asr, outputs

def ASR_calculate(dataset_path, reject_sentence_path, base_model_path,prefix=None):
    """
    CSVファイル (dataset_path) の 'input' 列をバッチ推論し，reject_sentence_path の拒否キーワードが
    出力に含まれていればrejectと判定．レイヤーごとに削除 → ASRを計算 → グラフ化
    推論には MetaLlama3 クラスを用いる．
    """
    # ==== 1) CSVファイルからプロンプトのリストを取得 ====
    df = pd.read_csv(dataset_path)  # 必要に応じてencoding指定
    prompts = df["input"].tolist()  # "input"列からリスト化
    prompts = prompts[:300]

    # ==== 2) reject_sentence.txt から拒否フレーズをロード ====
    reject_phrases = load_reject_phrases(reject_sentence_path)
    print("Reject phrases:", reject_phrases)

    # ==== 3) ベースモデル（Llama）をロード ====
    print("[INFO] Loading base LlamaForCausalLM...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path) 

    # レイヤー数
    total_layers = len(base_model.model.layers)
    del base_model
    torch.cuda.empty_cache()
    # remove_indices_list = [[layer_idx] for layer_idx in range(total_layers-1)]
    remove_indices_list = [[layer_idx] for layer_idx in range(total_layers)]
    generate_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "do_sample": True
    }

    results = {}
    model_name = base_model_path.split('/')[-1]
    lang = reject_sentence_path.split('_')[-2]
    result_dir = f"./{model_name}_{lang}_{prefix}"
    os.makedirs(result_dir,exist_ok=True)
    # ==== 4) レイヤーを一つずつ削除 → ASRを計算 ====
    for remove_list in remove_indices_list:
        print("\n===============================")
        print(f"[START] Remove layer indices = {remove_list}")
        
        # 4.3 MetaLlama3でロードし，バッチ推論 → ASR計算
        metal = MetaLlama3(model_path=base_model_path,abration_layer_list=remove_list)
        print(f"[COMPUTE] Compute ASR")
        asr,outputs = compute_asr_with_metal(
            model=metal,
            prompts=prompts,
            reject_phrases=reject_phrases,
            batch_size=50,
            generation_config=generate_kwargs
        )

        print(f"[RESULT] Remove {remove_list} => ASR = {asr:.4f}")
        results[str(remove_list)] = asr
        
        os.makedirs(result_dir,exist_ok=True)
        
        gen_df = pd.DataFrame({
            "index" : range(len(prompts)),
            "prompt": prompts,
            "generated_text":outputs
        })
        gen_df.to_csv(os.path.join(result_dir,f"generated_removed{str(remove_list)}.csv"),index=False)
        
        
        print(f"[SAVED] Generated to generated_removed{str(remove_list)}.csv")

        # メモリ解放
        del metal
        torch.cuda.empty_cache()

        print(f"[CLEANUP] Cuda Cache")

    # ==== 5) 可視化（例：棒グラフ） ====
    x = list(results.keys())
    y = list(results.values())

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, color='orange')
    plt.xlabel("Removed Layer Index",fontsize=12)
    plt.ylabel("ASR (Attack Success Rate)",fontsize=12)
    plt.grid(True,linestyle='--',alpha=0.7)
    plt.xticks(rotation=45, fontsize=10)

    plt.title(f"ASR by Removed Layer Indices {model_name} {lang}")
    plt.tight_layout()
    plt.savefig(f"asr_by_layer_removal_{model_name}_{lang}_{prefix}.png")

    print("\n=== Summary of Results ===")
    
    results_df = pd.DataFrame(list(results.items()), columns=["Removed Layers", "ASR"])
    results_df.to_csv(os.path.join(result_dir, f"./asr_results{model_name}_lang{lang}.csv"), index=False)
    for k, v in results.items():
        print(f"Removed {k} => ASR: {v:.4f}")

def main():
    ASR_calculate(
        dataset_path="../dataset_for_sahara/Multilingual_ja_300.csv",
        reject_sentence_path="./reject_keywords_ja_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf",
        prefix = "multilingual_final"
    )
    ASR_calculate(
        dataset_path="../dataset_for_sahara/Multilingual_en_300.csv",
        reject_sentence_path="./reject_keywords_en_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf",
        prefix = "multilingual_final"
    )
    ASR_calculate(
        dataset_path="./attack_dataset/adv_bench_en.csv",
        reject_sentence_path="./reject_keywords_en_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf",
        prefix = "adv_bench_final"
    )
    ASR_calculate(
        dataset_path="./attack_dataset/adv_bench_ja.csv",
        reject_sentence_path="./reject_keywords_ja_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf",
        prefix = "adv_bench_final"
    )
    # ASR_calculate(
    #     dataset_path="../dataset_for_sahara/Multilingual_ja_600.csv",
    #     reject_sentence_path="./reject_keywords_ja_.txt",
    #     base_model_path="../models/Llama-3.2-3B-Instruct"
    # )
    # ASR_calculate(
    #     dataset_path="../dataset_for_sahara/Multilingual_en_600.csv",
    #     reject_sentence_path="./reject_keywords_en_.txt",
    #     base_model_path="../models/Llama-3.2-3B-Instruct"
    # )

if __name__ == "__main__":
    main()
