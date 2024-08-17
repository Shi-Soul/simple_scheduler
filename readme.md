


`cd ~/2024-intern/people/wjxie/repo_v2; ~/anaconda3/envs/yq/bin/python -c "import torch;print(torch.cuda.device_count())"`

`cd ~/2024-intern/people/wjxie; git reset --hard HEAD; git pull`

```bash
EXPNAME=PLR_Se2 && cd ~/2024-intern/people/wjxie/v3 && ~/anaconda3/envs/yq/bin/python trainer.py --seed 2 --use_enchead_ver 4 --n_layer 12 --n_head 8 --block_size 8 --n_embd 256 --n_hidden 256 --use_agent_mask 0 --norm_position prenorm --use_len_ratio 1  --ucbgc_beta 1.0 --batch_size 1024 --max_grad_norm 200.0 --learning_rate 0.00003 --lr_min 0 --time_weight_loss none --lamb_ratio 0.1 --max_iters 200000 --eval_interval 1000 --save_interval 10000 --datapath /home/xieweiji/2024-intern/people/wjxie/Boston/databatch/100m_R1_n.pkl --setting boston --expname $EXPNAME -s 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 1 --ddp_world_size 1 --__use_blockjux 1 --grad_accumulation 1 && ~/anaconda3/envs/yq/bin/python eval.py -lmc 1 -mlf ./model/boston/$EXPNAME/final.pth



YQ_PYTHON=~/anaconda3/envs/yq/bin/python; $YQ_PYTHON -c "import torch;print(torch.cuda.device_count())"


cd ~/2024-intern/people/wjxie/v3 && ~/anaconda3/envs/yq/bin/python trainer.py --use_enchead_ver 4 --n_layer 20 --n_head 8 --block_size 64 --n_embd 256 --n_hidden 256 --use_agent_mask 0 --norm_position prenorm --use_len_ratio 0 --ucbgc_beta 1.5 --batch_size 512 --max_grad_norm 1.0 --learning_rate 0.001 --lr_min 0 --time_weight_loss none --max_iters 50000 --eval_interval 1000 --save_interval 10000 --datapath /home/xieweiji/2024-intern/people/wjxie/Boston/databatch/100m_R1_n.pkl --setting boston --expname STTEST_V4_L20Hi256 -s 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --__use_blockjux 1 --grad_accumulation 1 --expcomm "Exp on Hyp" && ~/anaconda3/envs/yq/bin/python eval.py -lmc 1 -mlf ./model/boston/STTEST_V4_L20Hi256/final.pth


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 &&  export EXPNAME=ReB8V5_S2 && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 8 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --datapath ../Jinan/real5s_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --seed 2 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 --expcomm "This One use shuffle val_dataloader, we wish the val_loss be more stable." &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth &> /dev/null


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 &&  export EXPNAME=SmTB8V5 && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 8 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --datapath ../Shenzhen/simu5s_80w.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --seed 2 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 --expcomm "This One use shuffle val_dataloader, we wish the val_loss be more stable." &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth &> /dev/null


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/SmB8V5_Hi128/final.pth &> /dev/null

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/ReB8V5/final.pth 

python eval.py -lmc 1 -mlf ./model/jinan/ReB4V5/final.pth

# ./model/jinan/SmB8V5_Hi256             
# ./model/jinan/ReB4V5

# ./model/jinan/SmB4V5_Hi512
# ./model/jinan/SmB4V5_Hi128
./model/jinan/SmB16V5_Hi128
./model/jinan/SmB8V5_Hi128



YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D100p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 448362 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 &> /dev/null
# [448362, 844, 1]

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D30p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 134509 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D10p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 44836 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 &> /dev/null


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D3p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 13451 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 &> /dev/null

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D1p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 4483 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 &> /dev/null






YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D100p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 698610 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 &> /dev/null
# [698610, 1500, 1]

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D30p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 209583 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D10p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 69861 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 &> /dev/null


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D3p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 20958 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 &> /dev/null

YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D1p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 6986 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 &> /dev/null




python eval.py -lmc 1 -mlf XXX -d ../Shenzhen/real_val_jux.pkl --batch_size 512 --_split_tvs 0

python eval.py -lmc 1 -mlf XXX -d ../Jinan/real5s_v2val_jux.pkl --batch_size 512 --_split_tvs 0

export CUDA_VISIBLE_DEVICES=6 && python eval.py -lmc 1 -mlf ./model/jinan/SmTB4V5/final.pth -d ../Jinan/real5s_v2val_jux.pkl --batch_size 512 --_split_tvs 0 --_eval_all 1 &
export CUDA_VISIBLE_DEVICES=7 && python eval.py -lmc 1 -mlf ./model/jinan/SmTB32V5/final.pth -d ../Jinan/real5s_v2val_jux.pkl --batch_size 512 --_split_tvs 0 &



YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D70p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 489027 --datapath ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl --setting shenzhen --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/shenzhen/$EXPNAME/final.pth --batch_size 512 &> /dev/null



YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && export EXPNAME=ReTB16V5_D50p && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 16 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 512 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --_split_tvs 0 --num_few_shots 224181 --datapath ../Jinan/real5s_v2train_jux.pkl,../Jinan/real5s_v2val_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 2 &> /dev/null && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth --batch_size 512 &> /dev/null



export CUDA_VISIBLE_DEVICES=3 &&  python trainer.py --is_ft 1 --use_kl_reg 1 --kl_reg_factor 1 --batch_size 512 --max_grad_norm 1000000 --learning_rate 1e-5 --max_iters 50000 --_split_tvs 0 -d ../Shenzhen/real_not_mask_train_jux.pkl,../Shenzhen/real_not_mask_val_jux.pkl -lmc 1 -mlf model/shenzhen/SmTB16V5/final.pth --expname OODFTTB16V5_KL1 -s 1 && python eval.py -lmc 1 -mlf model/shenzhen/OODFTTB16V5_KL1/final.pth -d ../Shenzhen/real_mask_tot_asval_jux.pkl --batch_size 512 --_split_tvs 0


export CUDA_VISIBLE_DEVICES=6 && export EXPNAME=FTTB16V5_KL0.01 &&  python trainer.py --is_ft 1 --use_kl_reg 1 --kl_reg_factor 0.01 --batch_size 512 --max_grad_norm 1000000 --learning_rate 1e-5 --max_iters 50000 --_split_tvs 0 -d ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl -lmc 1 -mlf model/shenzhen/SmTB16V5/final.pth --expname $EXPNAME -s 1 && python eval.py -lmc 1 -mlf model/shenzhen/$EXPNAME/final.pth -d ../Shenzhen/real_test_jux.pkl --batch_size 512 --_split_tvs 0

export CUDA_VISIBLE_DEVICES=7 && export EXPNAME=FTTB16V5_KL0.1 &&  python trainer.py --is_ft 1 --use_kl_reg 1 --kl_reg_factor 0.1 --batch_size 512 --max_grad_norm 1000000 --learning_rate 1e-5 --max_iters 50000 --_split_tvs 0 -d ../Shenzhen/real_train_jux.pkl,../Shenzhen/real_val_jux.pkl -lmc 1 -mlf model/shenzhen/SmTB16V5/final.pth --expname $EXPNAME -s 1 && python eval.py -lmc 1 -mlf model/shenzhen/$EXPNAME/final.pth -d ../Shenzhen/real_test_jux.pkl --batch_size 512 --_split_tvs 0





YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 && $YQ_PYTHON eval.py -lmc 1 -mlf ./model/jinan/ReTB16V5_D10p/final.pth -d ../Jinan/real5s_v2test_jux.pkl --_split_tvs 0 --batch_size 512 --_eval_all 1

```
