
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
    return f"GPU \n{gpu}\n{all}\n{use}\n{rest}"



def main():
    print_vram_info()
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")


    snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct",local_dir="./models/Llama-3.1-8B-Instruct",token = hf_token)


    from Batch_process_Sahara.attribution_parallel import safety_head_attribution_parallel

    import os
    default_search_cfg = {
        "search_step": 1,
        "mask_qkv": ['q'],
        "scale_factor": 1e-5,
        "mask_type": "scale_mask"
    }


    model_path = "./models/Llama-3.1-8B-Instruct"
    data_path = "./exp_data/maliciousinstruct.csv"
    storage_path = "./exp_res/sahara/Llama-3.1-8B-Instruct/"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    safety_head_attribution_parallel(
            model_name=model_path,
            data_path=data_path,
            search_cfg=default_search_cfg,
            storage_path=storage_path,
            device='cuda:0'
    )
    



if __name__=="__main__":
    try:
        main()
    except Exception as e:
        raise Exception(e)