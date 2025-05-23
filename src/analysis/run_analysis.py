import os
import argparse
from src.analysis.analyzer import OptimizerAnalyzer
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze optimization experimental results")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing workspace")
    parser.add_argument("--method", type=str, required=True, help="Optimization method (e.g., ga, pso)")
    parser.add_argument("--task_name", type=str, required=True, help="Task name(s) separated by underscore")
    parser.add_argument("--population_size", type=int, required=True, help="Population size used in optimization")
    parser.add_argument("--combine_method", type=str, required=True, help="Method used for combining weights")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save analysis results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = OptimizerAnalyzer(
        base_dir=args.base_dir,
        method=args.method,
        task_name=args.task_name,
        population_size=args.population_size,
        combine_method=args.combine_method,
        model=args.model
    )
    
    # Load states
    analyzer.load_states()
    
    # Plot convergence
    analyzer.plot_convergence(save_path=os.path.join(args.output_dir, "convergence.png"))
    
    # Analyze final performance
    final_stats = analyzer.analyze_final_performance()
    # Analyze and plot ensemble performance
    ensemble_stats = analyzer.analyze_ensemble_performance()
    analyzer.plot_ensemble_performance(save_path=os.path.join(args.output_dir, "ensemble_performance.png"))
    
    results_summary = {
        "final_performance": final_stats,
        "ensemble_performance": ensemble_stats
    }
    results_path = os.path.join(args.output_dir, "analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    # Print final performance statistics
    print("\nFinal Test Performance Statistics:")
    print("-" * 50)
    for metric, stats in final_stats.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
    
    
    
    # Print ensemble statistics
    print("\nEnsemble Performance Statistics:")
    print("-" * 50)
    for metric, data in ensemble_stats.items():
        print(f"\n{metric}:")
        best_k = data["best_k"]
        print(f"  Best Ensemble Size: {best_k['k']}")
        print(f"  Best Mean Score: {best_k['mean']:.4f}")
        print("\n  Detailed Statistics:")
        for top_k, stats in data["stats"].items():
            print(f"  {top_k}:")
            print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"    Max:  {stats['max']:.4f}")
            print(f"    Min:  {stats['min']:.4f}")
    
    # Analyze task correlation if multiple tasks
    if len(analyzer.tasks) > 1:
        analyzer.analyze_task_correlation(save_path=os.path.join(args.output_dir, "task_correlation.png"))


if __name__ == "__main__":
    main() 