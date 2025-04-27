python /home/szhang967/liftr/reasoning_analysis/compare_model_accuracy.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/circles_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    --labels "Claude-3-7-Sonnet" "DeepSeek-Reasoner" \
    --data_type classification \
    --output /home/szhang967/liftr/reasoning_analysis/compare_RLMs/circles \
    --unnormalized

python /home/szhang967/liftr/reasoning_analysis/compare_model_accuracy.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/blobs_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/blobs_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    --labels "Claude-3-7-Sonnet" "DeepSeek-Reasoner" \
    --data_type classification \
    --output /home/szhang967/liftr/reasoning_analysis/compare_RLMs/blobs \
    --unnormalized

python /home/szhang967/liftr/reasoning_analysis/compare_model_accuracy.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/linear_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/linear_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    --labels "Claude-3-7-Sonnet" "DeepSeek-Reasoner" \
    --data_type classification \
    --output /home/szhang967/liftr/reasoning_analysis/compare_RLMs/linear \
    --unnormalized

python /home/szhang967/liftr/reasoning_analysis/compare_model_accuracy.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/moons_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/moons_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_llm_analysis.json \
    --labels "Claude-3-7-Sonnet" "DeepSeek-Reasoner" \
    --data_type classification \
    --output /home/szhang967/liftr/reasoning_analysis/compare_RLMs/moons \
    --unnormalized
