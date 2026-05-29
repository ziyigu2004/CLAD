# Set the environment variables first before running the command.
# export HF_ENDPOINT=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model="Dream-org/Dream-v0-Instruct-7B"
model_name="Dream-v0-Instruct-7B"

device=0

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=1,block_length=256,add_bos_token=true,alg=klass,show_speed=True,temperature=0.2,top_p=0.95,conf_threshold=0.9,kl_threshold=0.001,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dawn,show_speed=True,conf_threshold=0.8,tau_induce=0.75,tau_sink=0.03,tau_edge=0.10,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dawn/${task}-ns0-${length}

# DAPD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dapd,show_speed=True,dapd_single_block=False,dapd_tau_min=0.001,dapd_tau_max=0.004,dapd_switch_ratio=0.5,dapd_fast_threshold=0.9,dapd_normalize_mask_graph=False,outp_path=evals_results_${model_name}/dapd/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dapd/${task}-ns0-${length}

# CLAD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=CLAD,threshold=0.75,show_speed=True,outp_path=evals_results_${model_name}/CLAD/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/CLAD/${task}-ns0-${length}

############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length}

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/parallel/${task}-ns0-${length}

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=1,block_length=256,add_bos_token=true,alg=klass,show_speed=True,temperature=0.2,top_p=0.95,conf_threshold=0.9,kl_threshold=0.005,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/klass/${task}-ns0-${length}

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dawn,show_speed=True,conf_threshold=0.8,tau_induce=0.75,tau_sink=0.03,tau_edge=0.10,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dawn/${task}-ns0-${length}

# DAPD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dapd,show_speed=True,dapd_single_block=False,dapd_tau_min=0.001,dapd_tau_max=0.004,dapd_switch_ratio=0.5,dapd_fast_threshold=0.9,dapd_normalize_mask_graph=False,outp_path=evals_results_${model_name}/dapd/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dapd/${task}-ns0-${length}

# CLAD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=CLAD,threshold=0.70,show_speed=True,outp_path=evals_results_${model_name}/CLAD/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/CLAD/${task}-ns0-${length}
    
############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=1,block_length=256,add_bos_token=true,alg=klass,show_speed=True,temperature=0.2,top_p=0.95,conf_threshold=0.8,kl_threshold=0.001,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/klass/${task}-ns0-${length} --log_samples

# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=dawn,show_speed=True,conf_threshold=0.8,tau_induce=0.75,tau_sink=0.03,tau_edge=0.10,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dawn/${task}-ns0-${length} --log_samples

# DAPD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dapd,show_speed=True,dapd_single_block=False,dapd_tau_min=0.0005,dapd_tau_max=0.0010,dapd_switch_ratio=0.5,dapd_fast_threshold=0.9,dapd_normalize_mask_graph=False,outp_path=evals_results_${model_name}/dapd/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dapd/${task}-ns0-${length} --log_samples

# CLAD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=CLAD,threshold=0.71,show_speed=True,outp_path=evals_results_${model_name}/CLAD/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/CLAD/${task}-ns0-${length} --log_samples

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=$((length / block_length))

# baseline
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results_${model_name}/baseline/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/baseline/${task}-ns0-${length} --log_samples

# parallel
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results_${model_name}/parallel/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/parallel/${task}-ns0-${length} --log_samples

# klass
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=1,block_length=256,add_bos_token=true,alg=klass,show_speed=True,temperature=0.2,top_p=0.95,conf_threshold=0.9,kl_threshold=0.001,outp_path=evals_results_${model_name}/klass/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/klass/${task}-ns0-${length} --log_samples
 
# dawn
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dawn,show_speed=True,conf_threshold=0.8,tau_induce=0.75,tau_sink=0.03,tau_edge=0.10,outp_path=evals_results_${model_name}/g-dllm/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dawn/${task}-ns0-${length} --log_samples

# DAPD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=dapd,show_speed=True,dapd_single_block=False,dapd_tau_min=0.0005,dapd_tau_max=0.0010,dapd_switch_ratio=0.5,dapd_fast_threshold=0.9,dapd_normalize_mask_graph=False,outp_path=evals_results_${model_name}/dapd/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/dapd/${task}-ns0-${length} --log_samples

# CLAD
CUDA_VISIBLE_DEVICES=${device} accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=CLAD,threshold=0.70,show_speed=True,outp_path=evals_results_${model_name}/CLAD/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code \
    --output_path evals_results_${model_name}/CLAD/${task}-ns0-${length} --log_samples