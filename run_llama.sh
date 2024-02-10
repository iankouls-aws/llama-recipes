torchrun --nproc_per_node=8 --nnodes=1 --rdzv_id=101 --rdzv_endpoint="localhost:59999" examples/finetuning.py --num_epochs=1 --dataset "samsum_dataset" --batch_size_training=1 --enable_fsdp --low_cpu_fsdp --model_name 'meta-llama/Llama-2-7b-hf' --output_dir .
