from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
import torch

def print_vram_info():
    if not torch.cuda.is_available():
        print("CUDAが有効になっていません。GPUが使用できる状態か確認してください。")
        return

    device = torch.device("cuda")
    total_vram = torch.cuda.get_device_properties(device).total_memory  # 全体のVRAM容量
    allocated_vram = torch.cuda.memory_allocated(device)  # 使用中のVRAM
    free_vram = total_vram - allocated_vram  # 空きVRAM

    gpu = f"GPU: {torch.cuda.get_device_name(device)}"
    all = f"全体のVRAM容量: {total_vram / 1024**3:.2f} GB"
    use = f"使用中のVRAM容量: {allocated_vram / 1024**3:.2f} GB"
    rest = (f"空きVRAM容量: {free_vram / 1024**3:.2f} GB")
    print(f"GPU \n{gpu}\n{all}\n{use}\n{rest}")



def main():
    print_vram_info()
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct",local_dir="./models/Llama-3.2-3B-Instruct",token = hf_token)
    snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct",local_dir="./models/Llama-3.1-8B-Instruct",token = hf_token)


    from lib.Sahara.attribution import safety_head_attribution

    import os
    default_search_cfg = {
        "search_step": 1,
        "mask_qkv": ['q'],
        "scale_factor": 1e-5,
        "mask_type": "mean_mask"
    }

    model_list = ["./models/Llama-3.2-3B-Instruct",
                    "./models/Llama-3.1-8B-Instruct"
                   ]
    dataset_list = [
        "./dataset_for_sahara/Multilingual.csv",
        "./dataset_for_sahara/Multilingual_de_300.csv",
        "./dataset_for_sahara/Multilingual_en_300.csv",
        "./dataset_for_sahara/Multilingual_hi_300.csv",
        "./dataset_for_sahara/Multilingual_ja_300.csv",
    ]
    
    storage_path = "./Result/"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    for data in dataset_list:
        for model in model_list:
            model_name = model.split('/')[-1]
            storage_path = os.path.join("./Result",model_name)
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
            safety_head_attribution(
                    model_name=model,
                    data_path=data,
                    search_cfg=default_search_cfg,
                    storage_path=storage_path,
                    device='cuda:0'
            )
    



if __name__=="__main__":
    main()
