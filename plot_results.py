import json
import matplotlib.pyplot as plt
import sys
from typing import Dict, List
import os

def plot_rouge_results(results_file: str, save_plot: bool = True) -> None:
    """
    Generate a line graph comparing ROUGE scores across different models.
    
    Args:
        results_file: Path to the JSON results file
        save_plot: Whether to save the plot as an image file
    """
    # Load results
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file '{results_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{results_file}'.")
        return

    # Extract data for plotting
    models = list(results["model_results"].keys())
    datasets = ["mlsum", "xlsum"]
    rouge_metrics = ["rouge1", "rouge2", "rougeL"]
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('ROUGE Score Comparison Across Models', fontsize=16, fontweight='bold')
    
    for dataset_idx, dataset in enumerate(datasets):
        ax = axes[dataset_idx]
        
        # Plot lines for each model
        for model_idx, model in enumerate(models):
            if model in results["model_results"] and dataset in results["model_results"][model]:
                scores = results["model_results"][model][dataset]["rouge_scores"]
                
                # Extract ROUGE scores
                rouge_values = [
                    scores["rouge1"],
                    scores["rouge2"], 
                    scores["rougeL"]
                ]
                
                # Plot line for this model
                ax.plot(rouge_metrics, rouge_values, 
                       marker='o', linewidth=2, markersize=8,
                       color=colors[model_idx % len(colors)],
                       label=model, alpha=0.8)
                
                # Add value labels on points
                for i, value in enumerate(rouge_values):
                    ax.annotate(f'{value:.3f}', 
                              (rouge_metrics[i], value),
                              textcoords="offset points",
                              xytext=(0,10), ha='center',
                              fontsize=9, alpha=0.7)
        
        # Customize subplot
        ax.set_title(f'{dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        ax.set_xlabel('ROUGE Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, max(0.6, ax.get_ylim()[1] + 0.05))  # Ensure readable scale
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Improve x-axis labels
        ax.set_xticks(range(len(rouge_metrics)))
        ax.set_xticklabels(['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        # Generate filename based on results file
        plot_filename = results_file.replace('.json', '_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    # Display plot
    plt.show()

def print_summary_stats(results_file: str) -> None:
    """Print summary statistics from the results."""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("ROUGE EVALUATION SUMMARY")
    print("="*60)
    print(f"Evaluation completed: {results['timestamp']}")
    print(f"Models tested: {len(results['model_results'])}")
    
    # Find best performing model for each metric
    best_models = {}
    
    for dataset in ["mlsum", "xlsum"]:
        print(f"\n{dataset.upper()} Dataset Results:")
        print("-" * 40)
        
        for metric in ["rouge1", "rouge2", "rougeL"]:
            best_score = 0
            best_model = ""
            
            for model in results["model_results"]:
                if dataset in results["model_results"][model]:
                    score = results["model_results"][model][dataset]["rouge_scores"][metric]
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            print(f"Best {metric.upper()}: {best_model} ({best_score:.4f})")
            best_models[f"{dataset}_{metric}"] = (best_model, best_score)
    
    print("\n" + "="*60)

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py <results_file.json>")
        print("Example: python plot_results.py rouge_evaluation_results_20250807_143022.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' does not exist.")
        sys.exit(1)
    
    # Print summary statistics
    print_summary_stats(results_file)
    
    # Generate plot
    plot_rouge_results(results_file)

if __name__ == "__main__":
    main()
