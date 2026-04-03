export CUDA_VISIBLE_DEVICES=0
mkdir -p different_lengths_logs
export PREFILL_CHUNK_SIZE=16384
export IMI_PREFETCH_MODE=random
export IMI_PREFETCH_SEED=42
# AdaptiveIMI with prefetch enabled
export IMI_PREFETCH=1
export IMI_PREFETCH_RATIO=0.1
CPU_MIN=0
CPU_MAX=95
RESULT_OUT=/data/yjy/RetrievalAttention/throughput_eval/result.csv
MODEL_NAMES=(
    Llama-3-8B-Instruct-Gradient-1048k
    # Qwen2.5-7B-Instruct
    # Llama-3.1-8B-Instruct
)


# ################################ Full Attention ################################
# for model_name in "${MODEL_NAMES[@]}"
# do
#     model_tag=$(basename "$model_name")
#     for context_len in 8000 16000 32000 64000 96000
#     do
#         for bsz in 1 
#         do
#             for round in 1 2 
#             do
#                 numactl --physcpubind=0-15 --membind=0 python -u test.py \
#                     --model_name $model_name \
#                     --attn_type Full_Flash_Attn \
#                     --context_len $context_len \
#                     --gen_len 128 \
#                     --task_name longbook_sum_eng_long \
#                     --truncate_input \
#                     --ignore_eos \
#                     --result_out ${RESULT_OUT} \
#                     --batch_size $bsz > different_lengths_logs/${model_tag}_full_attn_${context_len}_bsz${bsz}_${round}.log 2>&1
#             done
#         done
#     done
# done



# ################################ RetroInfer ################################
# for model_name in "${MODEL_NAMES[@]}"
# do
#     model_tag=$(basename "$model_name")
#     for context_len in 8000 16000 32000 64000 96000
#     do
#         for bsz in 1 
#         do
#             for round in 1 2 
#             do
#                 numactl --physcpubind=${CPU_MIN}-${CPU_MAX} --membind=all python -u test.py \
#                     --model_name $model_name \
#                     --attn_type RetroInfer \
#                     --use_cuda_graph \
#                     --retrieval_budget 0.1 \
#                     --estimation_budget 0 \
#                     --context_len $context_len \
#                     --gen_len 128 \
#                     --task_name longbook_sum_eng_long \
#                     --truncate_input \
#                     --ignore_eos \
#                     --result_out ${RESULT_OUT} \
#                     --batch_size $bsz > different_lengths_logs/${model_tag}_retroinfer_${context_len}_bsz${bsz}_${round}.log 2>&1
#             done
#         done
#     done
# done
for model_name in "${MODEL_NAMES[@]}"
do
    model_tag=$(basename "$model_name")
    for context_len in 240000
    do
        for bsz in 1 
        do
            for round in 1 2 
            do
                numactl --physcpubind=${CPU_MIN}-${CPU_MAX} --membind=all python -u test.py \
                    --model_name $model_name \
                    --attn_type RetroInfer \
                    --use_cuda_graph \
                    --context_len $context_len \
                    --gen_len 128 \
                    --task_name NIAH \
                    --truncate_input \
                    --ignore_eos \
                    --result_out ${RESULT_OUT} \
                    --batch_size $bsz > different_lengths_logs/${model_tag}_retroinfer_${context_len}_bsz${bsz}_${round}.log 2>&1
            done
        done
    done
done

# ################################ AdaptiveIMI ################################
# for model_name in "${MODEL_NAMES[@]}"
# do
#     model_tag=$(basename "$model_name")
#     # for context_len in 8000 16000 32000 64000 96000
#     for context_len in 240000
#     do
#         for round in 1 2 3
#         do
#             numactl --physcpubind=${CPU_MIN}-${CPU_MAX} --membind=all python -u test.py \
#                 --model_name $model_name \
#                 --attn_type AdaptiveIMI \
#                 --retrieval_budget 0.1 \
#                 --estimation_budget 0 \
#                 --context_len $context_len \
#                 --gen_len 128 \
#                 --task_name longbook_sum_eng_long \
#                 --truncate_input \
#                 --ignore_eos \
#                 --result_out ${RESULT_OUT} \
#                 --batch_size 1 \
#                 --subspace_parts 2 > different_lengths_logs/${model_tag}_adaptiveimi_${context_len}_bsz1_${round}.log 2>&1
#         done
#     done

# done


# ################################ Full_Flash_Attn_Offload ################################
# for model_name in "${MODEL_NAMES[@]}"
# do
#     model_tag=$(basename "$model_name")
#     for context_len in 8000 16000 32000 64000 96000
#     do
#         for round in 1 2 
#         do
#             numactl --physcpubind=${CPU_MIN}-${CPU_MAX} --membind=all python -u test.py \
#                 --model_name $model_name \
#                 --attn_type Full_Flash_Attn_Offload \
#                 --context_len $context_len \
#                 --gen_len 128 \
#                 --task_name longbook_sum_eng_long \
#                 --truncate_input \
#                 --ignore_eos \
#                 --result_out ${RESULT_OUT} \
#                 --batch_size 1 > different_lengths_logs/${model_tag}_full_attn_offload_${context_len}_bsz1_${round}.log 2>&1
#         done
#     done
# done


# ################################ AdaptiveIMI Decode Breakdown ################################
# bash throughput_eval/run_decode_breakdown.sh
unset IMI_PREFETCH
unset IMI_PREFETCH_RATIO
unset IMI_PROFILE_HIT_RATE
unset IMI_PROFILE_HIT_EVERY
unset CUDA_VISIBLE_DEVICES
