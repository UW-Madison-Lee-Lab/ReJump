#deepseek-reasoner
# regression
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/l1normreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/linreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/cosreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.02_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/pwreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/quadreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

# classification
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/linear_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/blobs_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/moons_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/expreg_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph



#claude-claude-3-7-sonnet-20250219-thinking
# regression
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/l1normreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/linreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/cosreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.02_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/pwreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/quadreg_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

# classification
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/circles_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/linear_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/blobs_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/moons_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph

python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/claude-claude-3-7-sonnet-20250219-thinking/expreg_50_shot_10_query_reasoning_api_reslen_3520_nsamples_500_noise_0.1_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0.3 \
    --max_tokens 25000 \
    --output_suffix _logical_graph






#gsm8k
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/gsm8k/claude/claude-thinking-gsm8k.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0 \
    --max_tokens 35000 \
    --output_suffix _logical_graph \
    --field_of_interests input+reasonings \
    --debug




python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/gsm8k/deepseek/deepseek-r1-gsm8k.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/logical_graph_prompt.txt \
    --llm gemini \
    --temperature 0 \
    --max_tokens 35000 \
    --output_suffix _logical_graph \
    --field_of_interests input+reasonings \
    --debug



###cognitive process graph
#/home/szhang967/liftr/reasoning_analysis/cognitive_process_graph_prompt.txt
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/gsm8k/deepseek/deepseek-r1-gsm8k.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/cognitive_process_graph_prompt.txt \
    --llm gemini \
    --temperature 0 \
    --max_tokens 35000 \
    --output_suffix _cognitive_process_graph \
    --field_of_interests input+reasonings \
    --debug

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/gsm8k/deepseek/deepseek-r1-gsm8k_gemini_analysis_debug_cognitive_process_graph.parquet


#for circles
python /home/szhang967/liftr/reasoning_analysis/analyze_responses.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default.parquet \
    --instruction /home/szhang967/liftr/reasoning_analysis/cognitive_process_graph_prompt.txt \
    --llm gemini \
    --temperature 0 \
    --max_tokens 35000 \
    --output_suffix _cognitive_process_graph \
    --field_of_interests input+reasonings \
    --debug

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_debug_cognitive_process_graph.parquet

####

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/gsm8k/deepseek/deepseek-r1-gsm8k_gemini_analysis_debug_logical_graph.parquet

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/gsm8k/claude/claude-thinking-gsm8k_gemini_analysis_logical_graph.parquet

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/circles_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_0.01_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_debug_logical_graph.parquet

python /home/szhang967/liftr/reasoning_analysis/llm_logical_graph_analysis.py \
    --input /home/szhang967/liftr/multi-query-results/deepseek-ai-deepseek-reasoner/blobs_50_shot_reasoning_api_reslen_3046_nsamples_500_noise_1.0_flip_rate_0.0_mode_default/global_step_0/test_default_gemini_analysis_debug_logical_graph.parquet