#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE     Input parquet file (default: deepseek-zeroshot.parquet)"
    echo "  -o, --output FILE    Output parquet file (optional, auto-generated if not provided)"
    echo "  -p, --instruction FILE  Instruction file (default: fitting_model_extraction_prompt.txt)"
    echo "  -l, --llm TYPE       LLM API to use: openai, claude, gemini, deepseek (default: openai)"
    echo "  -t, --temp VALUE     Temperature for generation (default: 0.3)"
    echo "  -m, --max VALUE      Maximum tokens for response (default: 1000)"
    echo "  -r, --retries NUM    Maximum retry attempts for API calls (default: 5)"
    echo "  -d, --delay VALUE    Delay between API calls in seconds (default: 1)"
    echo "  -s, --save-interval VALUE  How often to save results (every N rows, default: 1)"
    echo "  -c, --continue       Continue processing after errors"
    echo "  -v, --verbose        Enable verbose logging"
    echo "  --skip-test          Skip the testing phase"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  OPENAI_API_KEY       Required if using OpenAI API"
    echo "  ANTHROPIC_API_KEY    Required if using Claude API"
    echo "  GOOGLE_API_KEY       Required if using Gemini API"
    echo "  DEEPSEEK_API_KEY     Required if using DeepSeek API"
}

# Default values
INPUT_FILE="/home/szhang967/liftr/reasoning_analysis/deepseek-zeroshot.parquet"
OUTPUT_FILE=""
INSTRUCTION_FILE="/home/szhang967/liftr/reasoning_analysis/fitting_model_extraction_prompt.txt"
LLM_TYPE="openai"
TEMPERATURE=0.3
MAX_TOKENS=1000
MAX_RETRIES=5
DELAY=1
SAVE_INTERVAL=1
CONTINUE_ON_ERROR=""
VERBOSE=""
SKIP_TEST=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -p|--instruction)
            INSTRUCTION_FILE="$2"
            shift 2
            ;;
        -l|--llm)
            LLM_TYPE="$2"
            shift 2
            ;;
        -t|--temp)
            TEMPERATURE="$2"
            shift 2
            ;;
        -m|--max)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -r|--retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        -s|--save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        -c|--continue)
            CONTINUE_ON_ERROR="--continue-on-error"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --skip-test)
            SKIP_TEST=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Starting LLM analysis of responses..."

# Set working directory
cd /home/szhang967/liftr/reasoning_analysis

# Check if instruction file exists
if [ ! -f "$INSTRUCTION_FILE" ]; then
    echo "ERROR: Instruction file not found: $INSTRUCTION_FILE"
    exit 1
fi

# Check for required API key
case $LLM_TYPE in
    openai)
        if [ -z "$OPENAI_API_KEY" ]; then
            echo "ERROR: OPENAI_API_KEY environment variable is not set."
            echo "Please set it with: export OPENAI_API_KEY=your_key_here"
            exit 1
        fi
        ;;
    claude)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "ERROR: ANTHROPIC_API_KEY environment variable is not set."
            echo "Please set it with: export ANTHROPIC_API_KEY=your_key_here"
            exit 1
        fi
        ;;
    gemini)
        if [ -z "$GOOGLE_API_KEY" ]; then
            echo "ERROR: GOOGLE_API_KEY environment variable is not set."
            echo "Please set it with: export GOOGLE_API_KEY=your_key_here"
            exit 1
        fi
        ;;
    deepseek)
        if [ -z "$DEEPSEEK_API_KEY" ]; then
            echo "ERROR: DEEPSEEK_API_KEY environment variable is not set."
            echo "Please set it with: export DEEPSEEK_API_KEY=your_key_here"
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unsupported LLM type: $LLM_TYPE"
        echo "Supported types: openai, claude, gemini, deepseek"
        exit 1
        ;;
esac

# Run test script if not skipped
if [ $SKIP_TEST -eq 0 ]; then
    echo "Running tests..."
    python3 test_analyze_responses.py --input "$INPUT_FILE" --llm "$LLM_TYPE" $VERBOSE

    # Check if test was successful
    if [ $? -ne 0 ]; then
        echo "Tests failed. Please fix the issues before proceeding."
        exit 1
    fi
fi

# Build command with arguments
CMD="python3 analyze_responses.py --input \"$INPUT_FILE\" --llm \"$LLM_TYPE\" --instruction \"$INSTRUCTION_FILE\" --temperature $TEMPERATURE --max-tokens $MAX_TOKENS --max-retries $MAX_RETRIES --delay $DELAY --save-interval $SAVE_INTERVAL $CONTINUE_ON_ERROR $VERBOSE"

# Add output file if specified
if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\""
fi

# Run the main analysis script
echo -e "\nRunning analysis script..."
echo "Command: $CMD"
eval $CMD

echo "Analysis complete!" 