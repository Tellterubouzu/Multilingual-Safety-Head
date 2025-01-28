from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import torch
from tqdm import tqdm
class MetaLlama3:
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

        for i in tqdm(range(0, len(input_texts), batch_size),desc="Processing Batches"):
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

                generated_text = self.tokenizer.decode(
                    output_tokens[input_length:],
                    skip_special_tokens=True
                )

                results.append(generated_text)

        return results