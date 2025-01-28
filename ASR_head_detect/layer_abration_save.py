import torch
from transformers import LlamaForCausalLM

def remove_layers_from_llama(
    model_name_or_path: str,
    remove_layer_indices: list,
    output_dir: str = "./llama_modified"
):
    """
    Llama2モデルを読み込み，指定したレイヤーを削除して新しいモデルとして保存する関数．

    Args:
        model_name_or_path (str): 元となるLlama2モデルのパスや名称（例: "meta-llama/Llama-2-7b-hf"）
        remove_layer_indices (list): 削除したいレイヤーのインデックス（0-based）
        output_dir (str): 新しく作成したモデルを保存する先のディレクトリ
    """

    print(f"Loading original model from: {model_name_or_path}")
    model = LlamaForCausalLM.from_pretrained(model_name_or_path)

    original_layers = model.model.layers

    print(f"Original number of layers: {len(original_layers)}")

    remain_layers = torch.nn.ModuleList(
        layer for i, layer in enumerate(original_layers) 
        if i not in remove_layer_indices
    )
    model.model.layers = remain_layers

    model.config.num_hidden_layers = len(model.model.layers)
    print(f"Modified number of layers: {model.config.num_hidden_layers}")

    model.save_pretrained(output_dir)
    print(f"Modified model saved to: {output_dir}")


if __name__ == "__main__":
    remove_layer_indices_example = [0, 3]
    remove_layers_from_llama(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        remove_layer_indices=remove_layer_indices_example,
        output_dir="./llama2_modified"
    )
