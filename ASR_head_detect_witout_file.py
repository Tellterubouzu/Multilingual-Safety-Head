import os
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
from tqdm import tqdm
import pandas as pd
import os
import json
from lib.utils.custommodel import CustomLlamaModelForCausalLM
import shutil
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class CustomLlama:
    def __init__(self, model_path=None, quant_type=None, use_sys_prompt=False, sys_prompt="", generation_config=None):
        if model_path is None:
            model_path = "meta-llama/Llama3-3.1-8B-Instruct"
        
        self.model_name = model_path.split("/")[-1]
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" 

        if quant_type is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True if quant_type == 8 else False, load_in_4bit=True if quant_type == 4 else False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.default_generation_config = {
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 0.9,
            "max_new_tokens": 256
        }
        self.generation_config = generation_config or self.default_generation_config
        print("finish init operation")


    def predict_batch(self, input_texts, generation_config=None, batch_size=8):
        if not isinstance(input_texts, list):
            raise TypeError("predict_batch: input_texts must be a list of strings.")

        config = {**self.default_generation_config, **(generation_config or {})}
        results = []

        for i in range(0, len(input_texts), batch_size):
            batch_inputs = input_texts[i:i + batch_size]

            batch_prompts =[]
            for input_text in batch_inputs:
                local_chat = []
                local_chat.append({"role":"user","content":input_text})
                prompt = self.tokenizer.apply_chat_template(
                    local_chat,
                    tokenize = False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    top_p=config["top_p"],
                    temperature=config["temperature"],
                    max_new_tokens=config["max_new_tokens"],
                    eos_token_id=[
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    ],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for idx, output_tokens in enumerate(outputs):
                input_length = (inputs['attention_mask'][idx] == 1).sum().item()
                input_length = input_ids.shape[1]
                generated_text = self.tokenizer.decode(
                    output_tokens[input_length:],
                    skip_special_tokens=True
                ).strip()

                results.append(generated_text)

        return results



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

default_search_cfg = {
    "search_step": 1,
    "mask_qkv": ['q','k','v'],
    "scale_factor": 1e-5,
    "mask_type": "mean_mask"
}


def tuning_entity(layer:int, head:int):
    formatted  = {}
    formatted[(layer,head)] = ['q','k','v']
    text = f"L{layer}_H{head}"
    return formatted, text
def surgery(model, tokenizer,
            head_mask: dict = None,
            mask_type: str = None,
            scale_factor: float = None,
            path: str = None):
    temp_input = " "
    inputs = tokenizer.encode(temp_input, return_tensors='pt').to(model.device)
    outputs = model(inputs, head_mask=head_mask, mask_type=mask_type, scale_factor=scale_factor,
                    mask_para=True, head_dim=model.config.hidden_size // model.config.num_attention_heads)
    if path:
        tokenizer.save_pretrained(path)
        model.save_pretrained(path)
    del outputs
    return model

def head_tuning_and_save(base_model_path:str,layer:int,head:int):
    base_model_name = base_model_path.split("/")[-1]
    model = CustomLlamaModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    formatted,config_text = tuning_entity(layer,head)
    tuned_model_name = f"{base_model_name}-{config_text}"
    tuned_model_path= f"./temp_models/{base_model_name}-{config_text}"
    surgery_model = surgery(model = model,tokenizer = tokenizer,head_mask=formatted,mask_type = "mean_mask",path = tuned_model_path)
    
    del model, tokenizer, surgery_model
    torch.cuda.empty_cache()
    return tuned_model_path,tuned_model_name

def load_reject_phrases(file_path):
    if not os.path.exists(file_path):
        print(f"[WARN] {file_path} isn't exists. Return None List")
        return []
    with open(file_path, "r",encoding="utf-8")as f:
        phrases = [line.strip() for line in f if line.strip()]
    return phrases

def compute_asr_with_llama(model,prompts,reject_phrases,batch_size =4,generation_config=None):
    reject_count = 0

    outputs = model.predict_batch(prompts,generation_config,batch_size=batch_size)
    for text in outputs:
        if any(phrase in text for phrase in reject_phrases):
            reject_count +=1
    total  = len(prompts)
    asr = (1-(reject_count/total)) if total > 0 else 0.0
    return asr, outputs


def plot_layer_head_heatmap(file_path,base_model_name,lang):
    with open(file_path, 'r') as f:
        data = json.load(f)
    layer_head_scores = {}
    max_layer = 0
    max_head = 0

    for key, value in data.items():
        layer, head = map(int, key.split('-'))
        layer_head_scores[(layer, head)] = value
        max_layer = max(max_layer, layer)
        max_head = max(max_head, head)
    score_matrix = np.zeros((max_head + 1, max_layer + 1))

    for (layer, head), score in layer_head_scores.items():
        score_matrix[head, layer] = score
    score_matrix = np.flipud(score_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        score_matrix, 
        annot=False, 
        cmap="Blues", 
        xticklabels=[f"{i}" for i in range(max_layer + 1)],
        yticklabels=[f"{i}" for i in range(max_head, -1, -1)]
    )
    plt.xlabel("Layer Number")
    plt.ylabel("Head Number")
    plt.title(f"Layer-Head ASR Heatmap of {base_model_name}")
    plt.savefig(f"Layer-Head ASR Heatmap of {base_model_name},lang = {lang}")


def safety_head_attribution_by_asr(base_model_path,data_path,reject_sentence_path,lang:str,storage_path =None,device='cuda:0',batch_size=20,debug = True):
    
    df = pd.read_csv(data_path)

    prompts = df["input"].tolist()
    if debug:
        prompts = prompts[60:]
    base_model_name = base_model_path.split('/')[-1]
    results_dir = f"./ASR_head_results/{base_model_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(f"[Info] Evaluating {base_model_name}")
    
    reject_phrases = load_reject_phrases(reject_sentence_path)

    model = CustomLlamaModelForCausalLM.from_pretrained(base_model_path)
    layer_nums = model.config.num_hidden_layers
    head_nums = model.config.num_attention_heads
    del model
    torch.cuda.empty_cache()
    generation_kwargs = {
        "max_new_tokens":256,
        "temperature":0.9,
        "do_sample":True
    }
    results = {}

    
    for layer in tqdm(range(0,layer_nums)):
        for head in range(0,head_nums):
            # before_model = CustomLlamaModelForCausalLM.from_pretrained(base_model_path)
            # tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            # temp_dir = f"./L{layer}/H{head}"
            temp_dir = f"./L{layer}"

            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            tuned_model_path,tuned_model_name = head_tuning_and_save(
                base_model_path=base_model_path,
                layer=layer,
                head = head
            )
            print(f"[SAVED] Model{tuned_model_path}")
            model = CustomLlama(model_path = tuned_model_path)
            print(f"[Compute] Compute ASR")
            asr,outputs = compute_asr_with_llama(
                model=model,
                prompts=prompts,
                reject_phrases=reject_phrases,
                batch_size=batch_size,
                generation_config=generation_kwargs
            )
            print(f"[RESULT] Remove L{layer} H{head}=>ASR{asr}")
            results[f"{layer}-{head}"]=asr
            out_dir = os.path.join(results_dir,temp_dir)
            os.makedirs(out_dir,exist_ok=True)
            gen_df = pd.DataFrame({
                "index":range(len(prompts)),
                "prompt":prompts,
                "generated_text":outputs
            })
            gen_df.to_csv(os.path.join(out_dir,f"L{layer}_H{head}generated.csv"),index=False)
            print(f"[SAVED] Generated to {out_dir}")
            del tuned_model_name,tuned_model_path,model,asr,outputs,gen_df
            torch.cuda.empty_cache()
            shutil.rmtree(tuned_model_path,ignore_errors=True)
    result_file_path = os.path.join(results_dir,"asr_head_attribution.json")
    print(f"[Completed] Writing results at{results_dir}/asr_head.json")
    with open(result_file_path,'w',encoding='utf-8')as f:
        json.dump(results,f,ensure_ascii=False,indent =2)
    plot_layer_head_heatmap(result_file_path,base_model_name,lang)
    
            



def main():
    print_vram_info()
    import os
    safety_head_attribution_by_asr(
        data_path="../dataset_for_sahara/Multilingual_ja_600.csv",
        reject_sentence_path="./reject_keywords_ja_.txt",
        base_model_path="../models/Llama-3.1-8B-Instruct",
        batch_size=50,
        lang="ja"
    )
    safety_head_attribution_by_asr(
        data_path="../dataset_for_sahara/Multilingual_en_600.csv",
        reject_sentence_path="./reject_keywords_en_.txt",
        base_model_path="../models/Llama-3.1-8B-Instruct",
        batch_size=50,
        lang="en"
    )
    safety_head_attribution_by_asr(
        data_path="../dataset_for_sahara/Multilingual_ja_600.csv",
        reject_sentence_path="./reject_keywords_ja_.txt",
        base_model_path="../models/Llama-3.2-3B-Instruct",
        batch_size=50,
        lang="ja"
    )
    safety_head_attribution_by_asr(
        data_path="../dataset_for_sahara/Multilingual_en_600.csv",
        reject_sentence_path="./reject_keywords_en_.txt",
        base_model_path="../models/Llama-3.2-3B-Instruct",
        batch_size=50,
        lang="en"
    )
    



if __name__=="__main__":
    main()
