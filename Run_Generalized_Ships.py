
from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

try :
    from notify import send_email
except ImportError:
    pass

def main():
    send_email(body = "job activated")
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")


    snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct",local_dir="./models/Llama-3.1-8B-Instruct",token = hf_token)


    from lib.Sahara.attribution import safety_head_attribution

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

    safety_head_attribution(
            model_name=model_path,
            data_path=data_path,
            search_cfg=default_search_cfg,
            storage_path=storage_path,
            device='cuda:0'
    )
    send_email(body = "Job Finished")


if __name__=="__main__":
    try:
        main()
    except Exception as e:
        try:
            send_email(body= f"Error has Occurred /n/n/n{e}")
            raise Exception(e)
        except ImportError:
            raise Exception(e)