torchrun --nproc_per_node=8 --nnodes=1 examples/finetuning.py --num_epochs=1 --batch_size_training=3 --enable_fsdp --low_cpu_fsdp --model_name meta-llama/Llama-2-7b-hf --output_dir .
