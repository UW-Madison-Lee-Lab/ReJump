# regression
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/l1normreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/regression-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/linreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/regression-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/cosreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.02_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/regression-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/pwreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/regression-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/quadreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/regression-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

# classification
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/linear_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/blobs_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/moons_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000



###claude classification
# classification
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/circles_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/blobs_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/moons_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/linear_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/classification-fitting_model_extraction_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000