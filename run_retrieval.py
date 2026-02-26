"""
ICIR: Instance-level Composed Image Retrieval
Clean implementation of the retrieval pipeline.
"""

import numpy as np
import time
import os
import argparse
import csv

from utils import setup_device
from utils_features import load_model, read_icir
from utils_retrieval import calculate_rankings, map_calc_icir


# Method presets: predefined configurations for each retrieval method
METHOD_PRESETS = {
    "basic": {
        "description": "Full BASIC method with all components",
        "contextualize": True,
        "mahf": False,
        "qasp": False,
        "tgbqe": False,
        "specified_corpus": "generic_subjects",
        "specified_ncorpus": "generic_styles",
        "aa": 0.2,
        "qasp_beta": 1.0,
        "num_principal_components_for_projection": 250.0,
        "standardize_features": True,
        "use_laion_mean": True,
        "project_features": True,
        "do_query_expansion": True,
        "tgbqe_k": 25,
        "tgbqe_gamma": 50,
        "normalize_similarities": True,
        "path_to_synthetic_data": "./synthetic_data",
        "harris_lambda": 0.1,
        "mahf_mapping": "exp",
        "mahf_tau": 0.001,
        "mahf_lambda_min": 0.0,
        "mahf_lambda_max": 0.1
    },
    "sum": {
        "description": "Simple sum fusion of image and text similarities",
    },
    "product": {
        "description": "Simple product fusion of image and text similarities",
    },
    "image": {
        "description": "Image-only retrieval (no text)",
    },
    "text": {
        "description": "Text-only retrieval (no image)",
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Composed Image Retrieval")
    
    # Basic settings
    parser.add_argument("--gpu", default=0, type=int, help="GPU id")
    parser.add_argument("--dataset", default="icir", type=str, help="Dataset name")
    parser.add_argument("--backbone", choices=["clip", "siglip"], default="clip", type=str, help="Vision-language model backbone")
    parser.add_argument("--method", choices=["image", "text", "sum", "product", "basic"], type=str, default="basic", help="Retrieval method")
    
    # Text processing
    parser.add_argument("--contextualize", action="store_true", help="Contextualize text queries with corpus")
    parser.add_argument("--mahf", action="store_true", help="Enable Modality-Adaptive Harris Fusion modifier for basic")
    parser.add_argument("--qasp", action="store_true", help="Enable Query-Adaptive Dynamic Subspace Projection modifier for basic")
    parser.add_argument("--tgbqe", action="store_true", help="Enable Text-Guided Bimodal Query Expansion modifier for basic")
    
    # Decomposition method parameters
    parser.add_argument("--specified_corpus", type=str, default="generic_subjects", help="Positive corpus for PCA projection")
    parser.add_argument("--specified_ncorpus", type=str, default="generic_styles", help="Negative corpus for PCA projection")
    parser.add_argument("--aa", type=float, default=0.2, help="Negative corpus weight in contrastive PCA")
    parser.add_argument("--qasp_beta", type=float, default=2.0, help="QASP ReLU exponent (must be >=1)")
    parser.add_argument("--num_principal_components_for_projection", type=float, default=250.0, 
                        help="Number of PCA components (>1) or energy threshold (<1)")
    
    # Feature processing
    parser.add_argument("--standardize_features", action="store_true", help="Standardize features before projection")
    parser.add_argument("--use_laion_mean", action="store_true", help="Use pre-computed LAION mean for standardization")
    parser.add_argument("--project_features", action="store_true", help="Apply PCA projection")

    # Similarity refinement
    parser.add_argument("--do_query_expansion", action="store_true", help="Expand queries with top retrieved samples")
    parser.add_argument("--tgbqe_k", type=int, default=25, help="Number of TG-BQE expansion neighbors")
    parser.add_argument("--tgbqe_gamma", type=float, default=0.1, help="TG-BQE softmax temperature scale")
    parser.add_argument("--normalize_similarities", action="store_true", help="Min-max normalize similarities")
    parser.add_argument("--path_to_synthetic_data", type=str, default=None, help="Path to synthetic normalization data")
    parser.add_argument("--harris_lambda", type=float, default=0.1, help="Harris corner detection lambda for fusion")

    # MA-HF specific settings
    parser.add_argument("--mahf_mapping", choices=["exp", "sigmoid"], default="exp",
                        help="Adaptive penalty mapping for MA-HF")
    parser.add_argument("--mahf_tau", type=float, default=0.2,
                        help="Temperature for MA-HF adaptive mapping")
    parser.add_argument("--mahf_lambda_min", type=float, default=0.0,
                        help="Lower bound for MA-HF adaptive Harris penalty")
    parser.add_argument("--mahf_lambda_max", type=float, default=0.1,
                        help="Upper bound for MA-HF adaptive Harris penalty")
    
    # Output
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    
    # Ablation control
    parser.add_argument("--use_preset", action="store_true", 
                        help="Use method preset configuration (recommended). Disable to manually override all args.")
    
    return parser.parse_args()


def apply_method_preset(args):
    """
    Apply method preset configuration with selective overrides.
    
    If --use_preset is enabled (default behavior), load preset for the method.
    Command-line arguments can still override preset values for ablations.
    
    Args:
        args: Parsed arguments from argparse
    
    Returns:
        Updated args with preset applied
    """
    if not args.use_preset:
        print("⚠️  Preset disabled - using command-line arguments only")
        return args
    
    if args.method not in METHOD_PRESETS:
        print(f"⚠️  No preset found for method '{args.method}' - using command-line arguments")
        return args
    
    preset = METHOD_PRESETS[args.method]
    print(f"✓ Applying '{args.method}' preset: {preset['description']}")
    
    # Get parser to check which args were explicitly provided by user
    parser = argparse.ArgumentParser()
    # Re-add all arguments (needed to detect defaults)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dataset", choices=["icir"], default="icir", type=str)
    parser.add_argument("--backbone", choices=["clip", "siglip"], default="clip", type=str)
    parser.add_argument("--method", choices=["image", "text", "sum", "product", "basic"], type=str, default="basic")
    parser.add_argument("--contextualize", action="store_true")
    parser.add_argument("--mahf", action="store_true")
    parser.add_argument("--qasp", action="store_true")
    parser.add_argument("--tgbqe", action="store_true")
    parser.add_argument("--specified_corpus", type=str, default="generic_subjects")
    parser.add_argument("--specified_ncorpus", type=str, default="generic_styles")
    parser.add_argument("--aa", type=float, default=0.2)
    parser.add_argument("--qasp_beta", type=float, default=2.0)
    parser.add_argument("--num_principal_components_for_projection", type=float, default=250.0)
    parser.add_argument("--standardize_features", action="store_true")
    parser.add_argument("--use_laion_mean", action="store_true")
    parser.add_argument("--project_features", action="store_true")
    parser.add_argument("--do_query_expansion", action="store_true")
    parser.add_argument("--tgbqe_k", type=int, default=25)
    parser.add_argument("--tgbqe_gamma", type=float, default=0.1)
    parser.add_argument("--normalize_similarities", action="store_true")
    parser.add_argument("--path_to_synthetic_data", type=str, default=None)
    parser.add_argument("--harris_lambda", type=float, default=0.1)
    parser.add_argument("--mahf_mapping", choices=["exp", "sigmoid"], default="exp")
    parser.add_argument("--mahf_tau", type=float, default=0.2)
    parser.add_argument("--mahf_lambda_min", type=float, default=0.0)
    parser.add_argument("--mahf_lambda_max", type=float, default=0.1)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--use_preset", action="store_true")
    
    # Parse with just defaults to compare
    defaults = parser.parse_args([])
    
    # Apply preset values, but preserve user-provided overrides
    overrides = []
    for key, preset_value in preset.items():
        if key == "description":
            continue
            
        current_value = getattr(args, key)
        default_value = getattr(defaults, key)
        
        # If user explicitly set this arg (different from default), keep it as override
        if current_value != default_value:
            overrides.append(f"  • {key}: {preset_value} → {current_value} (user override)")
        else:
            # Use preset value
            setattr(args, key, preset_value)
    
    if overrides:
        print("  Overrides applied:")
        for override in overrides:
            print(override)
    
    return args


def build_method_variant(args):
    """
    Build output-safe method identifier including active basic modifiers.
    """
    if args.method.lower() != "basic":
        return args.method.lower()

    modifiers = []
    if args.mahf:
        modifiers.append("mahf")
    if args.qasp:
        modifiers.append("qasp")
    if args.tgbqe:
        modifiers.append("tgbqe")

    if not modifiers:
        return "basic"
    return "basic_" + "_".join(modifiers)



def load_dataset(backbone, dataset_name, device, contextualize, norm=True):
    """
    Load query and database features for icir dataset.
    
    Returns:
        Tuple of (data_dict, dataset_name) where data_dict has structure:
        {
            "query": {"image_feats", "text_feats", "paths", "instances", "texts"},
            "database": {"image_feats", "paths", "instances", "texts"}
        }
    """
    features_dir = os.path.join("features", f"{backbone}_features", dataset_name)
    query_path = os.path.join(features_dir, "query_icir_features.pkl")
    database_path = os.path.join(features_dir, "database_icir_features.pkl")
    
    print(f"Loading dataset from {features_dir}...")
    
    # Read features - returns structured dictionary with "query" and "database" keys
    data = read_icir(query_path, database_path, device, contextualize=contextualize, norm=norm)
    
    return data


def process_instance(instance, data, args):
    """
    Process a single instance: retrieve relevant queries/database items and compute rankings.
    
    Args:
        instance: Instance identifier (e.g., "homer_simpson")
        data: Dictionary containing query and database subdictionaries
        args: Command line arguments
        model: Vision-language model
        tokenizer: Text tokenizer
    
    Returns:
        Dictionary with results for this instance
    """
    # Get indices for this instance
    query_indices = [i for i, inst in enumerate(data["query"]["instances"]) if inst == instance]
    db_indices = [i for i, inst in enumerate(data["database"]["instances"]) if inst == instance]
    
    # Extract features for this instance
    query_img_feats = data["query"]["image_feats"][query_indices]
    query_txt_feats = data["query"]["text_feats"][query_indices]
    query_texts = [data["query"]["texts"][i] for i in query_indices]
    query_paths = [data["query"]["paths"][i] for i in query_indices]
    query_instances = [data["query"]["instances"][i] for i in query_indices]
    
    db_feats = data["database"]["image_feats"][db_indices]
    db_paths = [data["database"]["paths"][i] for i in db_indices]
    db_instances = [data["database"]["instances"][i] for i in db_indices]
    db_texts = [data["database"]["texts"][i] for i in db_indices]
    
    # Compute rankings
    rankings, aux = calculate_rankings(
        args=args,
        image_features=query_img_feats,
        text_features=query_txt_feats,
        database_features=db_feats,
        return_aux=True
    )
    
    # Calculate metrics
    metrics = map_calc_icir(
        rankings, db_paths, query_paths, query_instances, 
        query_texts, db_instances, db_texts
    )
    
    alignments = aux.get("alignment")
    lambda_qs = aux.get("lambda_q")
    qasp_active_negatives = aux.get("qasp_active_negatives")
    qasp_weight_entropy = aux.get("qasp_weight_entropy")
    qasp_max_weight = aux.get("qasp_max_weight")
    qasp_min_weight = aux.get("qasp_min_weight")
    qasp_num_positive_eigs = aux.get("qasp_num_positive_eigs")
    tgbqe_topk_mean_fusion = aux.get("tgbqe_topk_mean_fusion")
    tgbqe_anchor_weight = aux.get("tgbqe_anchor_weight")
    tgbqe_weight_entropy = aux.get("tgbqe_weight_entropy")
    if alignments is not None:
        alignments = alignments.tolist()
    if lambda_qs is not None:
        lambda_qs = lambda_qs.tolist()
    if qasp_active_negatives is not None:
        qasp_active_negatives = qasp_active_negatives.tolist()
    if qasp_weight_entropy is not None:
        qasp_weight_entropy = qasp_weight_entropy.tolist()
    if qasp_max_weight is not None:
        qasp_max_weight = qasp_max_weight.tolist()
    if qasp_min_weight is not None:
        qasp_min_weight = qasp_min_weight.tolist()
    if qasp_num_positive_eigs is not None:
        qasp_num_positive_eigs = qasp_num_positive_eigs.tolist()
    if tgbqe_topk_mean_fusion is not None:
        tgbqe_topk_mean_fusion = tgbqe_topk_mean_fusion.tolist()
    if tgbqe_anchor_weight is not None:
        tgbqe_anchor_weight = tgbqe_anchor_weight.tolist()
    if tgbqe_weight_entropy is not None:
        tgbqe_weight_entropy = tgbqe_weight_entropy.tolist()

    return {
        "rankings": rankings,
        "APs": metrics["APs"],
        "correct_matrix": metrics["correct_matrix"],
        "query_paths": query_paths,
        "query_texts": query_texts,
        "query_instances": query_instances,
        "db_paths": db_paths,
        "alignments": alignments,
        "lambda_qs": lambda_qs,
        "qasp_active_negatives": qasp_active_negatives,
        "qasp_weight_entropy": qasp_weight_entropy,
        "qasp_max_weight": qasp_max_weight,
        "qasp_min_weight": qasp_min_weight,
        "qasp_num_positive_eigs": qasp_num_positive_eigs,
        "tgbqe_topk_mean_fusion": tgbqe_topk_mean_fusion,
        "tgbqe_anchor_weight": tgbqe_anchor_weight,
        "tgbqe_weight_entropy": tgbqe_weight_entropy,
        "mean_AP": np.mean(metrics["APs"]),
        "min_AP": np.min(metrics["APs"])
    }


def build_query_log(instance_results, top_k=10):
    """
    Build detailed per-query log with top-k retrieval results.
    
    Args:
        instance_results: List of results dictionaries from process_instance
        top_k: Number of top results to include
    
    Returns:
        List of rows for CSV output
    """
    rows = []
    
    for result in instance_results:
        rankings = result["rankings"]
        correct_matrix = result["correct_matrix"]
        query_paths = result["query_paths"]
        query_texts = result["query_texts"]
        query_instances = result["query_instances"]
        db_paths = result["db_paths"]
        APs = result["APs"]
        
        for q_idx, ranking in enumerate(rankings):
            # Get top-k retrieved paths and match flags
            retrieved_paths = [db_paths[i] for i in ranking[:top_k]]
            match_flags = correct_matrix[q_idx][:top_k].cpu().numpy().astype(int).tolist()
            
            # Calculate metrics
            total_positives = correct_matrix[q_idx].cpu().numpy().sum()
            correct_matches = sum(match_flags)
            precision_at_k = correct_matches / top_k
            recall_at_k = correct_matches / (total_positives + 1e-8)
            ap = APs[q_idx]
            
            # Build row
            row = [
                query_paths[q_idx],
                query_texts[q_idx],
                round(ap, 4),
                round(precision_at_k, 4),
                round(recall_at_k, 4)
            ]
            
            # Add retrieved paths and match flags
            for path, match in zip(retrieved_paths, match_flags):
                row.extend([path, match])
            
            rows.append(row)
    
    return rows


def save_results(instance_results, args, method, dataset_name, elapsed_time):
    """
    Save all results to files.
    
    Args:
        instance_results: List of results from all instances
        args: Command line arguments
        method: Retrieval method name
        dataset_name: Dataset identifier
        elapsed_time: Total processing time
    """
    # Aggregate metrics
    all_APs = []
    mean_APs_per_instance = []
    min_APs_per_instance = []
    all_alignments = []
    all_lambda_qs = []
    all_qasp_active_negatives = []
    all_qasp_weight_entropy = []
    all_qasp_max_weight = []
    all_qasp_min_weight = []
    all_qasp_num_positive_eigs = []
    all_tgbqe_topk_mean_fusion = []
    all_tgbqe_anchor_weight = []
    all_tgbqe_weight_entropy = []
    is_mahf = args.method.lower() == "basic" and args.mahf
    is_qasp = args.method.lower() == "basic" and args.qasp
    is_tgbqe = args.method.lower() == "basic" and args.tgbqe
    
    for result in instance_results:
        all_APs.extend(result["APs"])
        mean_APs_per_instance.append(result["mean_AP"])
        min_APs_per_instance.append(result["min_AP"])
        if is_mahf and result["alignments"] is not None and result["lambda_qs"] is not None:
            all_alignments.extend(result["alignments"])
            all_lambda_qs.extend(result["lambda_qs"])
        if is_qasp and result["qasp_active_negatives"] is not None:
            all_qasp_active_negatives.extend(result["qasp_active_negatives"])
            all_qasp_weight_entropy.extend(result["qasp_weight_entropy"])
            all_qasp_max_weight.extend(result["qasp_max_weight"])
            all_qasp_min_weight.extend(result["qasp_min_weight"])
            all_qasp_num_positive_eigs.extend(result["qasp_num_positive_eigs"])
        if is_tgbqe and result["tgbqe_topk_mean_fusion"] is not None:
            all_tgbqe_topk_mean_fusion.extend(result["tgbqe_topk_mean_fusion"])
            all_tgbqe_anchor_weight.extend(result["tgbqe_anchor_weight"])
            all_tgbqe_weight_entropy.extend(result["tgbqe_weight_entropy"])
    
    mAP = round(np.mean(all_APs) * 100, 2)
    mmAP = round(np.mean(mean_APs_per_instance) * 100, 2)  # Mean of instance means
    minmAP = round(np.mean(min_APs_per_instance) * 100, 2)  # Mean of instance mins

    mahf_stats = {}
    if is_mahf and len(all_alignments) > 0:
        alignments_np = np.array(all_alignments, dtype=np.float32)
        lambda_qs_np = np.array(all_lambda_qs, dtype=np.float32)
        mahf_stats = {
            "alignment_mean": float(np.mean(alignments_np)),
            "alignment_std": float(np.std(alignments_np)),
            "alignment_min": float(np.min(alignments_np)),
            "alignment_max": float(np.max(alignments_np)),
            "lambda_mean": float(np.mean(lambda_qs_np)),
            "lambda_std": float(np.std(lambda_qs_np)),
            "lambda_min": float(np.min(lambda_qs_np)),
            "lambda_max": float(np.max(lambda_qs_np)),
        }

    qasp_stats = {}
    if is_qasp and len(all_qasp_active_negatives) > 0:
        active_np = np.array(all_qasp_active_negatives, dtype=np.float32)
        entropy_np = np.array(all_qasp_weight_entropy, dtype=np.float32)
        maxw_np = np.array(all_qasp_max_weight, dtype=np.float32)
        minw_np = np.array(all_qasp_min_weight, dtype=np.float32)
        eigs_np = np.array(all_qasp_num_positive_eigs, dtype=np.float32)
        qasp_stats = {
            "active_negatives_mean": float(np.mean(active_np)),
            "weight_entropy_mean": float(np.mean(entropy_np)),
            "max_weight_mean": float(np.mean(maxw_np)),
            "min_weight_mean": float(np.mean(minw_np)),
            "num_positive_eigs_mean": float(np.mean(eigs_np)),
        }

    tgbqe_stats = {}
    if is_tgbqe and len(all_tgbqe_topk_mean_fusion) > 0:
        topk_np = np.array(all_tgbqe_topk_mean_fusion, dtype=np.float32)
        anchor_np = np.array(all_tgbqe_anchor_weight, dtype=np.float32)
        entropy_np = np.array(all_tgbqe_weight_entropy, dtype=np.float32)
        tgbqe_stats = {
            "topk_mean_fusion_mean": float(np.mean(topk_np)),
            "anchor_weight_mean": float(np.mean(anchor_np)),
            "weight_entropy_mean": float(np.mean(entropy_np)),
        }
    
    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Method: {method}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Total queries: {len(all_APs)}")
    print(f"  Total instances: {len(instance_results)}")
    print(f"  mAP: {mAP}%")
    print(f"  mmAP (mean per instance): {mmAP}%")
    print(f"  minmAP (mean of min per instance): {minmAP}%")
    if mahf_stats:
        print(f"  MA-HF alignment mean/std/min/max: "
              f"{mahf_stats['alignment_mean']:.4f} / {mahf_stats['alignment_std']:.4f} / "
              f"{mahf_stats['alignment_min']:.4f} / {mahf_stats['alignment_max']:.4f}")
        print(f"  MA-HF lambda(q) mean/std/min/max: "
              f"{mahf_stats['lambda_mean']:.4f} / {mahf_stats['lambda_std']:.4f} / "
              f"{mahf_stats['lambda_min']:.4f} / {mahf_stats['lambda_max']:.4f}")
    if qasp_stats:
        print(f"  QASP active negatives (mean): {qasp_stats['active_negatives_mean']:.2f}")
        print(f"  QASP weight entropy (mean): {qasp_stats['weight_entropy_mean']:.4f}")
        print(f"  QASP max/min weight (mean): "
              f"{qasp_stats['max_weight_mean']:.4f} / {qasp_stats['min_weight_mean']:.4f}")
        print(f"  QASP positive eigenvalues (mean): {qasp_stats['num_positive_eigs_mean']:.2f}")
    if tgbqe_stats:
        print(f"  TG-BQE top-k fusion (mean): {tgbqe_stats['topk_mean_fusion_mean']:.4f}")
        print(f"  TG-BQE anchor weight (mean): {tgbqe_stats['anchor_weight_mean']:.4f}")
        print(f"  TG-BQE weight entropy (mean): {tgbqe_stats['weight_entropy_mean']:.4f}")
    print(f"  Time: {elapsed_time:.1f}s")
    print(f"{'='*60}\n")
    
    # Save mAP summary
    mAP_dir = os.path.join(args.results_dir, "mAP")
    os.makedirs(mAP_dir, exist_ok=True)
    mAP_file = os.path.join(mAP_dir, f"{args.backbone}_{dataset_name}_{method}.txt")
    
    with open(mAP_file, "w") as f:
        # First: Configuration
        f.write(f"{'='*60}\n")
        f.write(f"Configuration:\n")
        f.write(f"{'='*60}\n")
        args_dict = vars(args)
        
        # Exclude runtime-specific arguments
        excluded_keys = {'results_dir', 'gpu', 'device'}
        advanced_basic_keys = {
            'mahf', 'qasp', 'tgbqe', 'contextualize',
            'specified_corpus', 'specified_ncorpus', 'aa',
            'qasp_beta', 'num_principal_components_for_projection', 'standardize_features',
            'use_laion_mean', 'project_features', 'do_query_expansion',
            'tgbqe_k', 'tgbqe_gamma', 'normalize_similarities', 'path_to_synthetic_data',
            'harris_lambda', 'mahf_mapping', 'mahf_tau', 'mahf_lambda_min', 'mahf_lambda_max'
        }
        
        # For non-basic methods, also exclude basic-specific parameters
        if args.method.lower() != 'basic':
            excluded_keys.update(advanced_basic_keys)
        
        filtered_dict = {k: v for k, v in args_dict.items() if k not in excluded_keys}
        max_key_len = max(len(key) for key in filtered_dict.keys())
        for key, value in sorted(filtered_dict.items()):
            f.write(f"{key:<{max_key_len + 2}}: {value}\n")
        
        # Then: Results
        f.write(f"\n{'='*60}\n")
        f.write(f"Results:\n")
        f.write(f"{'='*60}\n")
        f.write(f"map: {mAP}\n")
        f.write(f"mmap: {mmAP}\n")
        f.write(f"minmap: {minmAP}\n")
        if mahf_stats:
            f.write(f"alignment_mean: {mahf_stats['alignment_mean']:.6f}\n")
            f.write(f"alignment_std: {mahf_stats['alignment_std']:.6f}\n")
            f.write(f"alignment_min: {mahf_stats['alignment_min']:.6f}\n")
            f.write(f"alignment_max: {mahf_stats['alignment_max']:.6f}\n")
            f.write(f"lambda_mean: {mahf_stats['lambda_mean']:.6f}\n")
            f.write(f"lambda_std: {mahf_stats['lambda_std']:.6f}\n")
            f.write(f"lambda_min: {mahf_stats['lambda_min']:.6f}\n")
            f.write(f"lambda_max: {mahf_stats['lambda_max']:.6f}\n")
        if qasp_stats:
            f.write(f"qasp_active_negatives_mean: {qasp_stats['active_negatives_mean']:.6f}\n")
            f.write(f"qasp_weight_entropy_mean: {qasp_stats['weight_entropy_mean']:.6f}\n")
            f.write(f"qasp_max_weight_mean: {qasp_stats['max_weight_mean']:.6f}\n")
            f.write(f"qasp_min_weight_mean: {qasp_stats['min_weight_mean']:.6f}\n")
            f.write(f"qasp_num_positive_eigs_mean: {qasp_stats['num_positive_eigs_mean']:.6f}\n")
        if tgbqe_stats:
            f.write(f"tgbqe_topk_mean_fusion_mean: {tgbqe_stats['topk_mean_fusion_mean']:.6f}\n")
            f.write(f"tgbqe_anchor_weight_mean: {tgbqe_stats['anchor_weight_mean']:.6f}\n")
            f.write(f"tgbqe_weight_entropy_mean: {tgbqe_stats['weight_entropy_mean']:.6f}\n")
    
    print(f"Saved mAP summary to: {mAP_file}")
    
    # Save per-query APs
    APs_dir = os.path.join(args.results_dir, "APs")
    os.makedirs(APs_dir, exist_ok=True)
    APs_file = os.path.join(APs_dir, f"{args.backbone}_{dataset_name}_{method}.csv")
    
    with open(APs_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Query", "Text Query", "AP"])
        
        for result in instance_results:
            for path, text, ap in zip(result["query_paths"], result["query_texts"], result["APs"]):
                writer.writerow([path, text, ap])
    
    print(f"Saved per-query APs to: {APs_file}")

    if is_mahf:
        mahf_dir = os.path.join(args.results_dir, "MAHF")
        os.makedirs(mahf_dir, exist_ok=True)
        mahf_file = os.path.join(mahf_dir, f"{args.backbone}_{dataset_name}_{method}.csv")
        with open(mahf_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_path", "text", "instance", "alignment", "lambda_q", "AP"])
            for result in instance_results:
                if result["alignments"] is None or result["lambda_qs"] is None:
                    continue
                for path, text, instance, alignment, lambda_q, ap in zip(
                    result["query_paths"],
                    result["query_texts"],
                    result["query_instances"],
                    result["alignments"],
                    result["lambda_qs"],
                    result["APs"],
                ):
                    writer.writerow([
                        path,
                        text,
                        instance,
                        round(float(alignment), 6),
                        round(float(lambda_q), 6),
                        float(ap),
                    ])
        print(f"Saved MA-HF query statistics to: {mahf_file}")

    if is_qasp:
        qasp_dir = os.path.join(args.results_dir, "QASP")
        os.makedirs(qasp_dir, exist_ok=True)
        qasp_file = os.path.join(qasp_dir, f"{args.backbone}_{dataset_name}_{method}.csv")
        with open(qasp_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_path", "text", "instance",
                "active_negatives", "weight_entropy", "max_weight", "min_weight",
                "num_positive_eigs", "AP"
            ])
            for result in instance_results:
                if result["qasp_active_negatives"] is None:
                    continue
                for row in zip(
                    result["query_paths"],
                    result["query_texts"],
                    result["query_instances"],
                    result["qasp_active_negatives"],
                    result["qasp_weight_entropy"],
                    result["qasp_max_weight"],
                    result["qasp_min_weight"],
                    result["qasp_num_positive_eigs"],
                    result["APs"],
                ):
                    path, text, instance, active, entropy, max_w, min_w, npos, ap = row
                    writer.writerow([
                        path, text, instance,
                        int(active), round(float(entropy), 6), round(float(max_w), 6), round(float(min_w), 6),
                        int(npos), float(ap)
                    ])
        print(f"Saved QASP query statistics to: {qasp_file}")

    if is_tgbqe:
        tgbqe_dir = os.path.join(args.results_dir, "TGBQE")
        os.makedirs(tgbqe_dir, exist_ok=True)
        tgbqe_file = os.path.join(tgbqe_dir, f"{args.backbone}_{dataset_name}_{method}.csv")
        with open(tgbqe_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_path", "text", "instance",
                "topk_mean_fusion", "anchor_weight", "weight_entropy", "AP"
            ])
            for result in instance_results:
                if result["tgbqe_topk_mean_fusion"] is None:
                    continue
                for row in zip(
                    result["query_paths"],
                    result["query_texts"],
                    result["query_instances"],
                    result["tgbqe_topk_mean_fusion"],
                    result["tgbqe_anchor_weight"],
                    result["tgbqe_weight_entropy"],
                    result["APs"],
                ):
                    path, text, instance, topk_mean, anchor, entropy, ap = row
                    writer.writerow([
                        path,
                        text,
                        instance,
                        round(float(topk_mean), 6),
                        round(float(anchor), 6),
                        round(float(entropy), 6),
                        float(ap),
                    ])
        print(f"Saved TG-BQE query statistics to: {tgbqe_file}")
    
    # Save detailed retrieval logs
    # Uncomment the following block to enable detailed logs
    # 
    # logs_dir = os.path.join(args.results_dir, "retrieval_logs")
    # os.makedirs(logs_dir, exist_ok=True)
    
    # top_k = 10
    # logs_file = os.path.join(logs_dir, f"{args.backbone}_{dataset_name}_{method}_top{top_k}_results.csv")
    
    # query_log_rows = build_query_log(instance_results, top_k=top_k)
    
    # with open(logs_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     header = ["Query Image", "Text Query", "AP", f"P@{top_k}", f"R@{top_k}"]
    #     header += [f"Top-{i+1} Path" for i in range(top_k)]
    #     header += [f"Top-{i+1} Match" for i in range(top_k)]
    #     writer.writerow(header)
    #     writer.writerows(query_log_rows)
    
    # print(f"Saved detailed retrieval logs to: {logs_file}")
    
    return mAP


def run_retrieval(args):
    """
    Main retrieval pipeline.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        mAP score
    """
    start_time = time.time()
    
    # Setup
    # typical setup here has all features pre-computed, including contextualization, so there is no need to load the model
    # model_struct = load_model(args.backbone, args.device)
    # model, tokenizer = model_struct["model"], model_struct["tokenizer"]
    
    # Load dataset (features are pre-normalized)
    data = load_dataset(args.backbone, args.dataset, args.device, 
                                     args.contextualize, norm=True)
    
    # Get unique instances
    unique_instances = sorted(set(data["query"]["instances"]))
    print(f"\nProcessing {len(unique_instances)} unique instances...")
    
    # Process each instance
    instance_results = []
    
    for idx, instance in enumerate(unique_instances):
        # Clear line and print progress (pad with spaces to clear previous text)
        progress = f"  [{idx+1}/{len(unique_instances)}] {instance}"
        print(f"{progress:<80}", end="\r")
        
        result = process_instance(instance, data, args)
        instance_results.append(result)
    
    print()  # New line after progress
    
    # Save results
    elapsed_time = time.time() - start_time
    method_variant = build_method_variant(args)
    mAP = save_results(instance_results, args, method_variant,
                      args.dataset, elapsed_time)
    
    return mAP


def main():
    """Entry point."""
    args = parse_args()
    
    # Apply method preset (can be overridden by command-line args)
    args = apply_method_preset(args)

    if args.method.lower() != "basic" and (args.mahf or args.qasp or args.tgbqe):
        raise ValueError("--mahf/--qasp/--tgbqe modifiers are only supported with --method basic")
    if args.method.lower() == "basic" and args.tgbqe and not args.do_query_expansion:
        raise ValueError("--tgbqe requires --do_query_expansion")
    if args.qasp_beta < 1.0:
        raise ValueError(f"--qasp_beta must be >= 1.0, got {args.qasp_beta}")
    if args.tgbqe_k <= 0:
        raise ValueError(f"--tgbqe_k must be > 0, got {args.tgbqe_k}")
    
    # Features are already normalized
    args.norm = True
    
    # Setup device
    args.device = setup_device(gpu_id=args.gpu)
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    args_dict = vars(args)
    max_key_len = max(len(key) for key in args_dict.keys())
    
    for key, value in sorted(args_dict.items()):
        print(f"  {key:<{max_key_len + 2}}: {value}")
    print("="*60 + "\n")
    
    # Run retrieval
    mAP = run_retrieval(args)
    
    print(f"\nFinal mAP: {mAP}%\n")


if __name__ == "__main__":
    main()
