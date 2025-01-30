import torch
import argparse
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
    #model = model
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, required=False)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path,trust_remote_code=True)
    peft_config_before = PeftConfig.from_pretrained(args.lora_model_name_or_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=peft_config_before.r, 
        lora_alpha=peft_config_before.lora_alpha, 
        lora_dropout=peft_config_before.lora_dropout,
        target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
    )
    #print("peft done")
    lora_model = get_peft_model(model, peft_config)
    lora_stat = torch.load(args.lora_model_name_or_path+"/adapter_model.bin")
    state_dict_renamed = {}
    for k, v in lora_stat.items():
        if "lora" in k and ".weight" in k:
            new_k = k.replace(".weight", ".default.weight")
            state_dict_renamed[new_k] = v
        else:
            state_dict_renamed[k] = v
    lora_model.load_state_dict(state_dict_renamed,strict=False)
    merge_model =lora_model.merge_and_unload()
    
    merge_model.half().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)