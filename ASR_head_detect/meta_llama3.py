import torch
from transformers import pipeline,BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig

class meta_llama3():
    def __init__(self, model_path=None,quant_type=None, use_sys_prompt=False, sys_prompt="", generation_config=None,use_history = False):
        if model_path==None:
            model_path="meta-llama/Llama3-3.1-8B-Instruct"
        self.model_name = model_path.split("/")[-1]
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token =self.tokenizer.eos_token
        self.use_history = use_history   
        if quant_type == None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map = "auto",
                torch_dtype = torch.bfloat16,
            )
        else:
            if quant_type==8:
                quantization_config=BitsAndBytesConfig(load_in_8bit=True)
            elif quant_type==4:
                quantization_config=BitsAndBytesConfig(load_in_4bit=True)
            else:
                raise Exception("Augment Error: quant_type must be 8 or 4 or None")
        
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype = torch.bfloat16,
                quantization_config = quantization_config,
                low_cpu_mem_usage = True,
                trust_remote_code = True,  
            )
        self.chat = []
        if use_sys_prompt:
            self.sys_prompt =sys_prompt
            self.make_prompt("system", sys_prompt)
        self.default_generation_config = {
            "do_sample": True,
            # "top_k": 50,
            "top_p": 0.90,
            "temperature": 0.8,
            "max_new_tokens": 4096
        }

        self.generation_config = generation_config if generation_config else self.default_generation_config

    def make_prompt(self, role, content):
        self.chat.append({"role": role, "content": content})

    def merge_configs(self, custom_config):
        config = self.default_generation_config.copy()
        if custom_config:
            config.update(custom_config)
        return config
    
    def predict(self, input, generation_config=None):
        config = self.merge_configs(generation_config)
        self.make_prompt("user", input)
        prompt = self.tokenizer.apply_chat_template(self.chat, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if config["do_sample"]==True:
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    do_sample=config["do_sample"],
                    # top_k=config["top_k"],
                    top_p=config["top_p"],
                    temperature=config["temperature"],
                    max_new_tokens=config["max_new_tokens"]
                )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    do_sample = False
                )
        generated = self.tokenizer.decode(outputs[0][len(input_ids['input_ids'][0]):], skip_special_tokens=True)
        if self.use_history == True:
            self.make_prompt("assistant",generated)
        else:        
            self.chat = []
        return generated
    def predict_batch(self,input_texts,generation_config =None):
        if not isinstance(input_texts,list):
            raise TypeError("prediction batch : The type of input_text require 'list'")
        config = self.merge_configs(generation_config)
        batch_prompts =[]
        for input_text in input_texts:
            local_chat = []
            if self.sys_prompt:
                local_chat.append({"role": "system", "content": self.sys_prompt})
            local_chat.append({"role": "user", "content": input_text})
            prompt = self.tokenizer.apply_chat_template(
                local_chat,
                tokenize=False,
                add_generation_prompt=True
            )
            batch_prompts.append(prompt)
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=config["do_sample"],
                top_p=config["top_p"],
                temperature=config["temperature"],
                max_new_tokens=config["max_new_tokens"]
            )
        generated_list = []
        for idx, output_tokens in enumerate(outputs):
            input_length = (inputs['attention_mask'][idx] == 1).sum().item()
            gen_text = self.tokenizer.decode(
                output_tokens[input_length:],
                skip_special_tokens=True
            )
            generated_list.append(gen_text)
        return generated_list
    def moderate_batch(self,inout_list):
        if not isinstance(inout_list,list):
            raise TypeError("moderate batch : The type of user_inputs require 'list")
        batch_history = []
        for inout in inout_list:
            history = []
            self.make_prompt(role = "user", content = inout[0])
            self.make_prompt()
            history.append({"role": "system", "content": input[0]})
            history.append({"role": "assistant", "content": input[1]})
            prompt = self.tokenizer.apply_chat_template(
                history,
                return_tensor = "pt"
            ).to(self.device)
            batch_history.append(prompt)
        inputs = self.tokenizer(
            batch_history,
            return_tensor="pt",
            padding = True
        )
        with torch.no_grad():
            moderate_result = self.model.generate(
                **inputs.
                max_new_tokens = 100,
                pad_token_id = 0
            )
        moderated_list =[]
        for idx, moderated in enumerate(moderate_result):
            prompt_len = input_ids.shape[-1]
            moderated = self.tokenizer.decode(moderate_result[0][prompt_len])



    def moderate(self,user_input,model_output):
        if "Guard"  not in self.model_name :
            raise PermissionError("If you call 'moderate' function, you must use 'Guard' model.")
        self.make_prompt(role = "user",content = user_input)
        self.make_prompt(role = "assistant", conent=model_output )
        input_ids = self.tokenizer.apply_chat_template(self.chat,return_tensor="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids = input_ids,max_new_tokens = 100, pad_token_id = 0)
        prompt_len = input_ids.shape[-1]
        self.chat = []
        return self.tokenizer.decode(output[0][prompt_len],skip_special_tokens = True)

# 使用例
if __name__ == "__main__":
    llama = meta_llama3(quant_type=None, use_sys_prompt=True, sys_prompt="You are a pirate chatbot who always responds in pirate speak!")
    response = llama.predict("Who are you?")
    print(response)
    # 単一推論
    prompt = "Hello, how are you today?"
    generated_single = llama.predict(prompt)
    print("[Single Prediction]")
    print("Prompt  :", prompt)
    print("Generated:", generated_single)

    print("####################################")
    # バッチ推論
    prompts = [
        "Hello, how are you today?",
        "Hello, how are you today?",
    ]
    generated_batch = llama.predict_batch(prompts)
    print("####################################")
    print("\n[Batch Prediction]")
    for inp, out in zip(prompts, generated_batch):
        print("Prompt  :", inp)
        print("Generated:", out)
        print("-----")
