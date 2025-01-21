import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
from peft import AutoPeftModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import argparse

def train( resultName, datasetSize ) :
    assert len(resultName) > 0, "There is an empty string for model name."
    assert datasetSize > 0, "Dataset size must be positive number."

    # Load a tokenizer to use its chat template
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    def format_prompt(example):
        """Format the prompt to using the <|user|> template TinyLLama is using"""

        chat = example["messages"]
        prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)

        return {"text": prompt}

    # Load and format the data using the template TinyLLama is using
    dataset = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        .shuffle(seed=42)
        .select(range(datasetSize))
    )
    dataset = dataset.map(format_prompt)
    dataset = dataset.remove_columns( ['messages'] )

    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Load the model to train on the GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"

    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

    # Prepare LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=32,  # LoRA Scaling
        lora_dropout=0.1,  # Dropout for LoRA Layers
        r=64,  # Rank
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=  # Layers to target
        ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"]
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)

    output_dir = "./results"

    args = SFTConfig(
            output_dir = output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            learning_rate = 2e-4,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            logging_steps=10,
            fp16=False,
            gradient_checkpointing=True,
            max_seq_length=512,
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,       
        peft_config=peft_config,
        args = args
    )

    trainer.train()

    # save lora weights
    trainer.model.save_pretrained(f"/cache/results/{resultName}")

def doInference( loraWeightsName, promtText ) :

    # Load a tokenizer to use its chat template
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"

    tokenizer.chat_template = template_tokenizer.chat_template

    model = AutoPeftModelForCausalLM.from_pretrained(
        f"/cache/results/{loraWeightsName}",
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    # merge lora weights
    merged_model = model.merge_and_unload()

    prompt = [{"role": "user", "content": promtText}]

    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=512, truncation=True)

    print( pipe(prompt)[0]["generated_text"] )



def main():
    parser = argparse.ArgumentParser(description="Script for training or inference of a model.")

    # General argument for operation mode
    parser.add_argument(
        '--mode', 
        choices=['training', 'inference'], 
        required=True, 
        help="Mode of operation: 'training' or 'inference'."
    )

    # General argument for model name
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help="Name of the model to use.",
        default="TinyLlama-1.1B-lora"
    )

    # Argument for training dataset size, only required if mode is 'training'
    parser.add_argument(
        '--dataset_size', 
        type=int, 
        required=False, 
        help="Size of the training dataset (required if mode is 'training')."
    )

    parser.add_argument(
        '--prompt', 
        type=str, 
        required=False, 
        default="Write joke about a chicken.",
        help="Promt for inference (required if mode is 'inference')."
    )

    args = parser.parse_args()

    if args.dataset_size is not None and args.dataset_size <= 0:
        parser.error("--dataset_size must be a positive number if specified.")

    if args.mode == 'inference' and len(args.prompt) == 0:
        parser.error("--prompt must be set if inference mode used.")

    if args.mode == 'training':
        if args.dataset_size is None:
            parser.error("--dataset_size is required when mode is 'training'.")
        print(f"Training mode selected.")
        print(f"Model: {args.model_name}")
        print(f"Dataset size: {args.dataset_size}")
        train( args.model_name, args.dataset_size )
    elif args.mode == 'inference':
        print(f"Inference mode selected.")
        print(f"Model: {args.model_name}")
        print(f"Prompt: {args.prompt}")
        doInference( args.model_name, args.prompt )

if __name__ == "__main__":
    main()
