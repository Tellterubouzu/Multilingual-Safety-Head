import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from accelerate import Accelerator
from lib.utils.get_model import get_model, get_tokenizer
from lib.Sahara.svd import compute_subspace_similarity, compute_subspace_spectral_norm

def load_from_search_cfg(search_cfg):
    temp = deepcopy(search_cfg)
    _ = temp.pop('search_step', None)
    return temp

def update_mask_cfg(mask_cfg, layer, head, temp=True):
    now_mask_key = (layer, head)
    new_mask_cfg = deepcopy(mask_cfg)
    if 'head_mask' not in new_mask_cfg:
        new_mask_cfg['head_mask'] = {}
    new_mask_cfg['head_mask'][now_mask_key] = mask_cfg['mask_qkv']
    return new_mask_cfg

def get_last_hidden_states(model, tokenizer, data, mask_cfg=None):
    with torch.no_grad():
        last_hidden_states = []
        head_mask = mask_cfg['head_mask'] if mask_cfg is not None else None
        mask_type = mask_cfg['mask_type'] if mask_cfg is not None else None
        scale_factor = mask_cfg['scale_factor'] if mask_cfg is not None else None
        
        for i, r in data.iterrows():
            input_text = f"## Query:{r['input']}\n## Answer:"
            inputs = tokenizer.encode(input_text, return_tensors='pt').to(model.model.device)
            outputs = model(inputs,
                            head_mask=head_mask,
                            mask_type=mask_type,
                            scale_factor=scale_factor,
                            output_hidden_states=True,
                           )
            now_lhs = outputs.hidden_states[-1]
            last_hidden_states.append(now_lhs[:, -1, :].reshape(-1))
        last_hidden_states = torch.stack(last_hidden_states)
        return last_hidden_states

def get_safety_subspace_shifts(base_last_hidden_states, last_hidden_states):
    shifts = compute_subspace_similarity(base_last_hidden_states, last_hidden_states)
    return shifts

def get_most_important_subspace(shifts_dict):
    """
    shifts_dict: {(layer, head): shift_val, ...}
    """
    sorted_dict = sorted(shifts_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_dict[0][0][0], sorted_dict[0][0][1]

def safety_head_attribution_parallel(
    model_name, 
    data_path, 
    storage_path=None, 
    search_cfg=None, 
    device='cuda'
):
    """
    Accelerate を用いて task parallel で (layer, head) を分割し、
    1ノード内で複数GPUに振り分けて並列実行する。
    """
    accelerator = Accelerator()  # Accelerateの初期化
    rank = accelerator.process_index       # 0,1,2,... のプロセスID
    world_size = accelerator.num_processes # 全プロセス数 (GPU数)

    # ランク0だけが表示するようにするヘルパー
    def log(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    log(f"[Rank {rank}] Starting up. world_size={world_size}")

    # モデルとトークナイザの読み込み（全プロセスで同一モデルをロード）
    tokenizer, _as = get_tokenizer(model_name)
    model, _acc = get_model(model_name, get_custom=True, add_size=False)
    
    # 使うデバイスをAccelerateが自動で割り当て
    # model は model.model と model.lm_head を持っているという構造なので両方 to(device)
    device_ = accelerator.device
    model.model.to(device_)
    model.lm_head.to(device_)

    # データ読み込み
    data = pd.read_csv(data_path)

    layer_nums = model.config.num_hidden_layers
    head_nums = model.config.num_attention_heads

    # search_cfg の設定を分解
    if search_cfg is None:
        search_cfg = {}
    search_step = search_cfg.get('search_step', 1)
    mask_cfg = load_from_search_cfg(search_cfg)  # "mask_qkv", "mask_type", "scale_factor" など

    # ベースの hidden states を1回だけ取得 (全プロセス同じデータに対して同じ結果)
    # ただし計算負荷が大きい場合、rank=0だけが計算してそれをブロードキャストしても良い
    base_lhs = get_last_hidden_states(model, tokenizer, data)

    # all_lhs を保存する場合のみ使う (storage_pathがNoneでなければ使う)
    if storage_path is not None:
        all_lhs = {}
    else:
        all_lhs = None

    # search_step だけ繰り返す
    for step in range(search_step):
        log(f"\n[Rank {rank}] ======= search_step: {step} =======")

        # 全 (layer, head) のリスト
        all_tasks = [(l, h) for l in range(layer_nums) for h in range(head_nums)]
        
        # (layer, head) を world_size で分割
        # rank毎に異なる部分だけを担当する
        chunk_size = (len(all_tasks) + world_size - 1) // world_size
        start_i = rank * chunk_size
        end_i = min(start_i + chunk_size, len(all_tasks))
        my_tasks = all_tasks[start_i:end_i]

        # ここで自分の担当分を計算する
        local_shifts_dict = {}
        for (layer, head) in my_tasks:
            # 既にマスク済みであればスキップ
            if 'head_mask' in mask_cfg and (layer, head) in mask_cfg['head_mask']:
                continue

            now_mask_cfg = update_mask_cfg(mask_cfg, layer, head)
            last_hs = get_last_hidden_states(model, tokenizer, data, now_mask_cfg)

            if all_lhs is not None:
                # CPUに退避して保存
                all_lhs[(layer, head)] = last_hs.detach().cpu()
            
            # shifts = get_safety_subspace_shifts(base_lhs, last_hs)
            shifts = compute_subspace_spectral_norm(base_lhs, last_hs)

            local_shifts_dict[(layer, head)] = shifts

        # 各プロセスが求めた shifts_dict を gather し、ランク0が統合
        # 辞書をそのまま集めるのは面倒なので、(layer, head, shift_val) のリストにして gather
        local_items = [(k[0], k[1], v) for k, v in local_shifts_dict.items()]
        gathered = accelerator.gather(local_items)  
        # gatherするとすべてのランクから結合リストが返る (同じ長さになるようにパディングされる可能性あり)
        # パディングされた(None)相当のデータは自分が持っていない部分なので無視

        if rank == 0:
            # ランク0だけが統合して辞書に戻す
            shifts_dict = {}
            for item in gathered:
                # item = (layer, head, shift_val)
                # ただし、集まった中には「パディングされたもの」が入る場合がある
                # 例えば (0,0,0.0) のようにダミーで使われることがあるので、適宜弾く
                layer_, head_, shift_val_ = item
                # チェック (分かりやすくするため、layer_とhead_が-1の時はパディングとする、など)
                # ここでは簡単のため、layer_numsやhead_numsの範囲かどうかで判定
                if 0 <= layer_ < layer_nums and 0 <= head_ < head_nums:
                    shifts_dict[(layer_, head_)] = float(shift_val_)
            
            # best_layer, best_head を決定
            if len(shifts_dict) == 0:
                # 全部マスク済みであればもう打ち切り
                log("[Rank 0] No tasks to update. Breaking.")
                break
            best_layer, best_head = get_most_important_subspace(shifts_dict)
            log(f"[Rank 0] best_layer={best_layer}, best_head={best_head}, shift_val={shifts_dict[(best_layer,best_head)]}")

        else:
            # ランク0以外はダミー変数を用意
            best_layer, best_head = None, None

        # ランク0で求めた (best_layer, best_head) を全プロセスにbroadcast
        best_layer = accelerator.broadcast(torch.tensor(best_layer if rank==0 else -1, dtype=torch.long), src=0)
        best_head  = accelerator.broadcast(torch.tensor(best_head if rank==0 else -1, dtype=torch.long), src=0)
        # Tensor -> int
        best_layer = best_layer.item()
        best_head = best_head.item()

        # 次のstepに向けて mask_cfg を更新
        # 全プロセスで同じmask_cfgになるようにする
        mask_cfg = update_mask_cfg(mask_cfg, best_layer, best_head, temp=False)

        # もし all_lhs を保存したい場合、ランク0で書き出し
        if storage_path is not None and rank == 0:
            # まず hidden states をまとめて保存 (今回のstep分だけでもよい)
            torch.save(all_lhs, f"{storage_path}/{os.path.basename(data_path)}_{step}.pt")

            # shifts_dict も保存
            # shifts_dict は (layer, head) => shift_val
            new_shifts_dict = {f"{k[0]}-{k[1]}": v for k, v in shifts_dict.items()}
            with open(f"{storage_path}/{os.path.basename(data_path)}_{step}.jsonl", "w+") as shifts_file:
                shifts_file.write(json.dumps(new_shifts_dict) + "\n")

    log(f"[Rank {rank}] Done")


# ==== メイン実行部。accelerate launch で呼ばれる想定 ====
if __name__ == '__main__':
    # 例: Llama-2-7b-chat のパス、CSVのパスを仮置き
    model_path = "./SafetyHeadAttribution/Llama-2-7b-chat-hf"
    data_path = "./SafetyHeadAttribution/exp_data/maliciousinstruct.csv"
    storage_path = "./SafetyHeadAttribution/exp_res/sahara"

    default_search_cfg = {
        "search_step": 1,
        "mask_qkv": ['q'],
        "scale_factor": 1e-5,
        "mask_type": "scale_mask"
    }

    # 出力先のディレクトリがなければ作成
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    safety_head_attribution_parallel(
        model_name=model_path,
        data_path=data_path,
        search_cfg=default_search_cfg,
        storage_path=storage_path,
        device='cuda'
    )

## accelerate launch --num_processes=8 --multi_gpu task_parallel_accelerate.py
