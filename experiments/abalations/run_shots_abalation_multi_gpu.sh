#!/bin/bash
# Create output directory
mkdir -p ../../output/ctx_length_ablation_plip
# Define a function to wait until specified GPU is available
wait_for_gpu() {
  local gpu_id=$1
  while true; do
    # Check GPU memory usage
    used_memory=$(nvidia-smi -i $gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$used_memory" -lt 10000 ]; then # Less than 10GB usage
      break
    fi
    echo "GPU $gpu_id is busy (${used_memory}MB used). Waiting for it to become available..."
    sleep 30 # Wait 30 seconds before checking again
  done
}
# Function to run ablation on a specific dataset using a specific GPU
run_ablation_on_dataset() {
  local gpu_id=$1
  local dataset=$2
  
  wait_for_gpu $gpu_id
  
  echo "Starting CTX length ablation for $dataset on GPU $gpu_id at $(date)"
  CUDA_VISIBLE_DEVICES=$gpu_id python ctx_length_ablation.py --dataset $dataset
  echo "Completed CTX length ablation for $dataset on GPU $gpu_id at $(date)"
}
echo "================================="
echo "PLIP CTX Length Ablation Study (Multi-GPU)"
echo "================================="
echo "Starting at $(date)"
echo "This study will distribute experiments across GPU 0, GPU 1, and GPU 2"
# Run each dataset on a separate GPU in parallel
run_ablation_on_dataset 0 "kather" &
run_ablation_on_dataset 1 "pannuke" &
run_ablation_on_dataset 2 "digestpath" &
# Wait for all parallel processes to complete
wait
echo "All CTX length ablation experiments completed at $(date)!"
# Run a final pass to make sure all combined summaries are generated
echo "Generating final combined summaries..."
python ctx_length_ablation.py --skip-plots --skip-latex
# Print summary
echo "Experiment Summary:"
echo "==================="
echo "CTX Length Ablation Using GPUs 0, 1, and 2"
echo ""
echo "Model: PLIP only"
echo ""
echo "Datasets:"
echo "- Kather (GPU 0)"
echo "- PanNuke (GPU 1)"
echo "- DigestPath (GPU 2)"
echo ""
echo "Fixed shots: 8"
echo ""
echo "CTX lengths tested: 2, 4, 6, 8, 10, 12, 14, 16, 32, 64"
echo ""
echo "Loss functions:"
echo "- CE+SLMDCA"
echo "- CE+SLMDCA+Cosine"
echo "- FL+SLMDCA"
echo "- FL+SLMDCA+Cosine"  
echo "- LS+SLMDCA"
echo "- LS+SLMDCA+Cosine"
echo ""
echo "Results can be found in: ../../output/ctx_length_ablation_plip/"
echo "- Individual results: ../../output/ctx_length_ablation_plip/*_seed*_result.csv"
echo "- Dataset results: ../../output/ctx_length_ablation_plip/*_all_ctx_results.csv"
echo "- Summaries: ../../output/ctx_length_ablation_plip/*_summary.csv"
echo "- Combined summary: ../../output/ctx_length_ablation_plip/combined_plip_summary.csv"
echo "- Pivot tables: ../../output/ctx_length_ablation_plip/*_ctx_pivot.csv"
echo "- Plots: ../../output/ctx_length_ablation_plip/plots/"
echo "- LaTeX tables: ../../output/ctx_length_ablation_plip/latex/"
# Calculate total runtime
start_time=$(date -d "$(grep "Starting at" $0 | head -1 | cut -d' ' -f3-)" +%s)
end_time=$(date +%s)
runtime=$((end_time - start_time))
hours=$((runtime / 3600))
minutes=$(( (runtime % 3600) / 60 ))
seconds=$((runtime % 60))
echo ""
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"