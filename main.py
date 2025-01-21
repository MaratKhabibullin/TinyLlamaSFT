def mainFunc() :
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Load a tokenizer to use its chat template
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    def format_prompt(example):
        """Format the prompt to using the <|user|> template TinyLLama is using"""

        # Format answers
        chat = example["messages"]
        prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)

        return {"text": prompt}

    # Load and format the data using the template TinyLLama is using
    dataset = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        .shuffle(seed=42)
        .select(range(1000))
    )
    dataset = dataset.map(format_prompt)
    dataset = dataset.remove_columns( ['messages'] )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig

    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # 4-bit quantization configuration - Q in QLoRA
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # Use 4-bit precision model loading
    #     bnb_4bit_quant_type="nf4",  # Quantization type
    #     bnb_4bit_compute_dtype="float16",  # Compute dtype
    #     bnb_4bit_use_double_quant=False,  # Apply nested quantization
    # )

    # Load the model to train on the GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",

        # Leave this out for regular SFT
        #quantization_config=bnb_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"
    #tokenizer.chat_template = template_tokenizer.chat_template

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

    # Prepare model for training
    #model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)

    from trl import SFTTrainer, SFTConfig

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
            
            #warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            # #max_steps = 60,
            # fp16 = not is_bfloat16_supported(),
            # bf16 = is_bfloat16_supported(),
            # optim = "adamw_8bit",
            # weight_decay = 0.01,
            # lr_scheduler_type = "linear",
            # seed = 3407,
            # report_to = "none",
            # max_seq_length = 2048,
            # dataset_num_proc = 4,
            # packing = False, # Can make training 5x faster for short sequences.
        )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        #dataset_text_field="text",
        tokenizer=tokenizer,
    # args=training_arguments,
        

        # Leave this out for regular SFT
        peft_config=peft_config,

        args = args
    )

    # Train model
    trainer.train()

    # Save QLoRA weights
    trainer.model.save_pretrained("/cache/results/TinyLlama-1.1B-lora")

def testModel() :
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

  # Load a tokenizer to use its chat template
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"

    tokenizer.chat_template = template_tokenizer.chat_template

    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        "/cache/results/TinyLlama-1.1B-lora",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    from transformers import pipeline

    # Use our predefined prompt template
    prompt = [{"role": "user", "content": "как дела?"}]
    #prompt = r'<|user|>\nWhat is ocean? Answer using 5 words.</s>\n<|assistant|>\n'
    #prompt = "<|user|> What is ocean? Answer using 5 words.</s><|assistant|>"

    # Run our instruction-tuned model
    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=512, truncation=True)
    result = pipe(prompt)[0]["generated_text"]

    print( result )


def testModelOld() :
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

  # Load a tokenizer to use its chat template
    template_tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "<PAD>"
    tokenizer.padding_side = "left"

    tokenizer.chat_template = template_tokenizer.chat_template

    from peft import AutoPeftModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    # Merge LoRA and base model
    merged_model = model

    from transformers import pipeline

    # Use our predefined prompt template
    prompt = [{"role": "user", "content": "Tell me something abount LLM."}]
    #prompt = r'<|user|>\nWhat is ocean? Answer using 5 words.</s>\n<|assistant|>\n'
    #prompt = "<|user|> What is ocean? Answer using 5 words.</s><|assistant|>"

    # Run our instruction-tuned model
    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=64, truncation=True)
    result = pipe(prompt)[0]["generated_text"]

    print( result )

if __name__=="__main__":
    #mainFunc()
    testModel()