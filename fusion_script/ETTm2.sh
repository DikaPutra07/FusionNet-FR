# =========== seq_96_pred_96 =============

python -u run_fussionnet.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model B6iFast \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --stride 8 \
  --kernel_size 25 \
  --des 'normal_0' \
  --n_learner 3 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --global_freq_pred 1 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --period_list 96 48 \
  --lradj type3 \
  --checkpoints ./checkpoints/

# =========== seq_96_pred_192 =============

python -u run_fussionnet.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model B6iFast \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --stride 8 \
  --kernel_size 25 \
  --des 'normal_0' \
  --n_learner 3 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --global_freq_pred 1 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --period_list 96 48 \
  --lradj type3 \
  --checkpoints ./checkpoints/

# =========== seq_96_pred_96 =============

python -u run_fussionnet.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model B6iFast \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --stride 8 \
  --kernel_size 25 \
  --des 'normal_0' \
  --n_learner 3 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --global_freq_pred 1 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --period_list 96 48 \
  --lradj type3 \
  --checkpoints ./checkpoints/

# =========== seq_96_pred_96 =============

python -u run_fussionnet.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model B6iFast \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --stride 8 \
  --kernel_size 25 \
  --des 'normal_0' \
  --n_learner 3 \
  --learning_rate 0.0001 \
  --d_model 128 \
  --d_ff 128 \
  --global_freq_pred 1 \
  --itr 1 \
  --fix_seed 2021 \
  --train_epochs 10 \
  --batch_size 32 \
  --patience 3 \
  --period_list 96 48 \
  --lradj type3 \
  --checkpoints ./checkpoints/