# =========== IOI ============
# dir_name="ioi"
# python train.py --probed_task ioi --save_dir $dir_name --batch_size 128 --num_epoch 100 --data_per_epoch 200000 --num_test_rollout 200 > ./data_and_model/ioi.txt

# python cache_generation.py --probed_task ioi

# =========== 3 digit Addition ============
# dir_name="addition"
# python train.py --probed_task addition --save_dir $dir_name --batch_size 256 --num_epoch 100 --data_per_epoch 1000000 --num_test_rollout 200 > ./data_and_model/addition.txt

# python cache_generation.py --probed_task addition

# =========== char counting ============

# dir_name="counting"
# python train.py --probed_task counting --rebalance 6.0 --save_dir $dir_name --batch_size 256 --num_epoch 100 --data_per_epoch 1000000 --num_test_rollout 200 > ./data_and_model/counting.txt

# python cache_generation.py --probed_task counting

# =========== factual recall ============
# dir_name="fact"
# python train.py --probed_task fact --save_dir $dir_name --batch_size 128 --acc_steps 1 --num_epoch 160 --data_per_epoch 200000 --num_test_rollout 100 --pretrained > ./data_and_model/fact.txt

# python cache_generation.py --probed_task fact