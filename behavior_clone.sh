python -m pdb -c continue cs285/scripts/run_hw1.py \
    --num_agent_train_steps_per_iter 1000 \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1