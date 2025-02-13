
# Run training
python3 run_SST2.py \
        --max_seq_length 256 \
        --num_train_epochs 15.0 \
        --do_train \
        --gpu_ids="0" \
        --gradient_accumulation_steps 8 \
        --print_step 100 \
        --early_stop 10000 \
        --train_batch_size 128

# Run evaluation in test set
python3 run_SST2.py \
        --max_seq_length 256 \
        --num_train_epochs 15.0 \
        --gpu_ids="0" \
        --gradient_accumulation_steps 8 \
        --print_step 100 \
        --early_stop 10000 \
        --train_batch_size 128
