


`cd ~/2024-intern/people/wjxie/repo_v2; ~/anaconda3/envs/yq/bin/python -c "import torch;print(torch.cuda.device_count())"`

`cd ~/2024-intern/people/wjxie; git reset --hard HEAD; git pull`

```bash
EXPNAME=PLR_Se2 && cd ~/2024-intern/people/wjxie/v3 && ~/anaconda3/envs/yq/bin/python trainer.py --seed 2 --use_enchead_ver 4 --n_layer 12 --n_head 8 --block_size 8 --n_embd 256 --n_hidden 256 --use_agent_mask 0 --norm_position prenorm --use_len_ratio 1  --ucbgc_beta 1.0 --batch_size 1024 --max_grad_norm 200.0 --learning_rate 0.00003 --lr_min 0 --time_weight_loss none --lamb_ratio 0.1 --max_iters 200000 --eval_interval 1000 --save_interval 10000 --datapath /home/xieweiji/2024-intern/people/wjxie/Boston/databatch/100m_R1_n.pkl --setting boston --expname $EXPNAME -s 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 1 --ddp_world_size 1 --__use_blockjux 1 --grad_accumulation 1 && ~/anaconda3/envs/yq/bin/python eval.py -lmc 1 -mlf ./model/boston/$EXPNAME/final.pth



YQ_PYTHON=~/anaconda3/envs/yq/bin/python; $YQ_PYTHON -c "import torch;print(torch.cuda.device_count())"


cd ~/2024-intern/people/wjxie/v3 && ~/anaconda3/envs/yq/bin/python trainer.py --use_enchead_ver 4 --n_layer 20 --n_head 8 --block_size 64 --n_embd 256 --n_hidden 256 --use_agent_mask 0 --norm_position prenorm --use_len_ratio 0 --ucbgc_beta 1.5 --batch_size 512 --max_grad_norm 1.0 --learning_rate 0.001 --lr_min 0 --time_weight_loss none --max_iters 50000 --eval_interval 1000 --save_interval 10000 --datapath /home/xieweiji/2024-intern/people/wjxie/Boston/databatch/100m_R1_n.pkl --setting boston --expname STTEST_V4_L20Hi256 -s 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --__use_blockjux 1 --grad_accumulation 1 --expcomm "Exp on Hyp" && ~/anaconda3/envs/yq/bin/python eval.py -lmc 1 -mlf ./model/boston/STTEST_V4_L20Hi256/final.pth


YQ_PYTHON=~/anaconda3/envs/yq/bin/python && cd ~/2024-intern/people/wjxie/v3 &&  export EXPNAME=SmB8V5_ga4 && $YQ_PYTHON trainer.py --use_enchead_ver 5 --n_layer 12 --n_head 8 --block_size 8 --n_embd 384 --n_hidden 384 --use_len_ratio 0 --ucbgc_beta 1.0 --batch_size 256 --max_grad_norm 100000.0 --learning_rate 0.0001 --max_iters 300000 --datapath ../Jinan/simu5s_50w_jux.pkl --setting jinan --expname $EXPNAME --short_runname 1 --use_wandb 1 --debug 0 --save_model 1 --use_dp 0 --ddp_world_size 1 --grad_accumulation 4 &> /dev/null && python eval.py -lmc 1 -mlf ./model/jinan/$EXPNAME/final.pth &> /dev/null


```
