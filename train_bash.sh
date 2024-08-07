accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11130 \
    --num_processes 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    --multi_gpu \
    train.py \
    --save_steps 5000 \
    --train_batch_size 2 \
    --data_json_file "./JSON_train.json" \
    --data_root_path "./resize_IMG" \
    --faceid_root_path "./all_faceID"  \
    --parsing_root_path "./parsing_mask_IMG" 
