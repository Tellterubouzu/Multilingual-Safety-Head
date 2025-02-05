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

def remove_layers_from_llama(base_model: AutoModelForCausalLM, remove_layer_indices: list):
    """
    既にロード済みの base_model (LlamaForCausalLM) から
    指定したレイヤー (remove_layer_indices) を削除した新しいモデルを返す．
    """
    
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
    """
    MetaLlama3インスタンスを使ってバッチ推論し，ASRを計算する関数．
    生成結果に reject_phrases のいずれかが含まれたら "reject" としてカウント．
    Attach Success Rateを返す
    """
    outputs = metal.predict_batch(prompts, generation_config=generation_config, batch_size=batch_size)
    reject_count = 0
    for text in outputs:
        # 拒否フレーズのどれかが含まれていたらreject
        if any(phrase in text for phrase in reject_phrases):
            reject_count += 1
    total = len(prompts)
    asr = ( 1-(reject_count / total)) if total> 0 else 0.0
    return asr, outputs

def ASR_base(dataset_path, reject_sentence_path, base_model_path):
    """
    CSVファイル (dataset_path) の 'input' 列をバッチ推論し，reject_sentence_path の拒否キーワードが
    出力に含まれていればrejectと判定．レイヤーごとに削除 → ASRを計算 → グラフ化
    推論には MetaLlama3 クラスを用いる．
    """

    # ==== 1) CSVファイルからプロンプトのリストを取得 ====
    df = pd.read_csv(dataset_path)  # 必要に応じてencoding指定
    prompts = df["input"].tolist()  # "input"列からリスト化

    # ==== 2) reject_sentence.txt から拒否フレーズをロード ====
    reject_phrases = load_reject_phrases(reject_sentence_path)
    print("Reject phrases:", reject_phrases)

    # ==== 3) ベースモデル（Llama）をロード ====
    print("[INFO] Loading base LlamaForCausalLM...")
    generate_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.9,
        "do_sample": True
    }

    model_name = base_model_path.split('/')[-1]
    lang = reject_sentence_path.split('_')[-2]
    result_dir = f"./{model_name}_{lang}"
    os.makedirs(result_dir,exist_ok=True)
    metal = MetaLlama3(model_path=base_model_path)
    print(f"[COMPUTE] Compute ASR")
    asr,outputs = compute_asr_with_metal(
        metal=metal,
        prompts=prompts,
        reject_phrases=reject_phrases,
        batch_size=10,
        generation_config=generate_kwargs
    )

    print(f"[RESULT] ASR = {asr:.4f}")
    os.makedirs(result_dir,exist_ok=True)
        
    gen_df = pd.DataFrame({
        "index" : range(len(prompts)),
        "prompt": prompts,
        "generated_text":outputs
    })
    gen_df.to_csv(os.path.join(result_dir,"generated.csv"),index=False) 
    print(f"[SAVED] Generated to {result_dir}")
    del metal
    torch.cuda.empty_cache()


def main():
    ASR_base(
        dataset_path="./attack_dataset/adv_bench_en.csv",
        reject_sentence_path="./reject_keywords_en_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf"
    )
    ASR_base(
        dataset_path="./attack_dataset/adv_bench_ja.csv",
        reject_sentence_path="./reject_keywords_ja_.txt",
        base_model_path="../models/Llama-2-7b-chat-hf"
    )

if __name__ == "__main__":
    main()
