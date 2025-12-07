#!/bin/bash
# Safe SGLang server launcher with memory optimizations for RunAI pods
# This script helps prevent OOM (exit code 137) errors

set -e  # Exit on error

# Default values
MODEL="${MODEL:-OpenMeditron/Meditron3-8B}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31000}"
MEM_FRACTION="${MEM_FRACTION:-0.6}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-2048}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model-path)
            MODEL="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --mem-fraction-static)
            MEM_FRACTION="$2"
            shift 2
            ;;
        --context-length|--max-model-len)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --max-total-tokens|--max-num-batched-tokens)
            MAX_TOTAL_TOKENS="$2"
            shift 2
            ;;
        --max-running-requests|--max-num-seqs)
            MAX_RUNNING_REQUESTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-path MODEL               Model path (default: $MODEL)"
            echo "  --dtype DTYPE                    Data type (default: $DTYPE)"
            echo "  --tensor-parallel-size SIZE      Tensor parallel size (default: $TP_SIZE)"
            echo "  --port PORT                      Server port (default: $PORT)"
            echo "  --mem-fraction-static FRACTION   Memory fraction for KV cache (default: $MEM_FRACTION)"
            echo "  --context-length LEN             Maximum context length (default: $CONTEXT_LENGTH)"
            echo "  --max-total-tokens TOKENS        Max total tokens (default: $MAX_TOTAL_TOKENS)"
            echo "  --max-running-requests REQS      Max running requests (default: $MAX_RUNNING_REQUESTS)"
            echo "  --help                           Show this help message"
            echo ""
            echo "Environment variables can also be used: MODEL, DTYPE, TP_SIZE, PORT,"
            echo "MEM_FRACTION, CONTEXT_LENGTH, MAX_TOTAL_TOKENS, MAX_RUNNING_REQUESTS"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "SGLang Server Configuration (Memory-Safe)"
echo "=========================================="
echo "Model:                  $MODEL"
echo "Data Type:              $DTYPE"
echo "Tensor Parallel:        $TP_SIZE"
echo "Port:                   $PORT"
echo "Memory Fraction:        $MEM_FRACTION"
echo "Context Length:         $CONTEXT_LENGTH"
echo "Max Total Tokens:       $MAX_TOTAL_TOKENS"
echo "Max Running Requests:   $MAX_RUNNING_REQUESTS"
echo "=========================================="
echo ""

# Check memory before starting
echo "Checking system memory..."
if command -v free &> /dev/null; then
    free -h
fi

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits | \
        awk '{printf "  GPU %d: %d/%d MB used (%.1f%%), %d MB free\n", NR-1, $1, $2, ($1/$2)*100, $3}'
fi

echo ""
echo "Starting SGLang server..."
echo ""

# Build the command with correct flag names
CMD="python3 -m sglang.launch_server \
    --model-path $MODEL \
    --dtype $DTYPE \
    --tensor-parallel-size $TP_SIZE \
    --port $PORT \
    --mem-fraction-static $MEM_FRACTION \
    --context-length $CONTEXT_LENGTH \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --max-running-requests $MAX_RUNNING_REQUESTS"

# Execute the command
echo "Executing: $CMD"
echo ""
exec $CMD

