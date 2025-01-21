# Repository Overview

This repository provides an example of training and inference using a LoRA (Low-Rank Adaptation) network for the **TinyLlama** model. The implementation runs inside a **Docker** container, making setup and execution straightforward. **Visual Studio Code** is recommended for debugging.

---

## Features

1. **Modes of Operation**  
   - **Fine-tuning**: Train the model with the **HuggingFaceH4/ultrachat_200k** dataset.  
   - **Inference**: Use the fine-tuned model to generate responses based on input prompts.

2. **Dockerized Environment**  
   All dependencies, including the model and dataset, are managed within a Docker container.

---

## Setup and Execution

### Build the Docker Image
```bash
docker build -t <docker image name> <this repo local path>
```

### Run Inference
Use the following command to run inference with a fine-tuned model:  
```bash
docker run -v <cache directory>:/cache:rw <docker image name> python main.py --mode inference --model_name <your model name> --prompt "what is love?"
```
- `<cache directory>`: Path to the folder for caching datasets and models.  
- `<docker image name>`: Name of the built Docker image.  
- `<your model name>`: Name of the fine-tuned model (e.g., `TinyLlama-1.1B-lora`).  

---

This repository demonstrates how to fine-tune and deploy lightweight language models using LoRA, making it an efficient solution for training and inference workflows.
