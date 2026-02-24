import torch
import numpy as np
import os
import pickle
import torch.nn.functional as F
from utils import *
from utils_features import *

def _normalized_cosine_alignment(image_features, text_features, eps=1e-8):
    """
    Compute per-query cosine alignment and normalize to [0, 1].

    Args:
        image_features: Tensor (num_queries, dim)
        text_features: Tensor (num_queries, dim)
        eps: Numerical stability epsilon

    Returns:
        alignment: Cosine alignment in [-1, 1]
        alignment_01: Alignment mapped to [0, 1]
    """
    image_features = image_features / image_features.norm(dim=1, keepdim=True).clamp_min(eps)
    text_features = text_features / text_features.norm(dim=1, keepdim=True).clamp_min(eps)
    alignment = (image_features * text_features).sum(dim=1).clamp(-1.0, 1.0)
    alignment_01 = (alignment + 1.0) / 2.0
    return alignment, alignment_01


def _adaptive_harris_lambda(args, alignment_01, eps=1e-8):
    """
    Compute query-adaptive Harris penalty for MA-HF.
    """
    tau = max(float(args.mahf_tau), eps)
    a01 = alignment_01.clamp(0.0, 1.0)

    if args.mahf_mapping == "exp":
        exp_floor = torch.exp(torch.tensor(-1.0 / tau, device=a01.device, dtype=a01.dtype))
        numerator = torch.exp((a01 - 1.0) / tau) - exp_floor
        denominator = (1.0 - exp_floor).clamp_min(eps)
        gain = (numerator / denominator).clamp(0.0, 1.0)
    else:  # sigmoid
        gain = torch.sigmoid((a01 - 0.5) / tau)

    lambda_q = args.mahf_lambda_min + (args.mahf_lambda_max - args.mahf_lambda_min) * gain
    return lambda_q


def apply_tg_bqe(
    q_v, q_t, X_v, P, s_min_v, s_min_t, lambda_val, k, gamma=0.1, return_stats=False, eps=1e-8
):
    """
    Text-Guided Bimodal Query Expansion (TG-BQE).

    Args:
        q_v: Centered visual query, shape (d,)
        q_t: Centered textual query, shape (d,)
        X_v: Centered database visual features, shape (N, d)
        P: Projection basis, shape (d, c)
        s_min_v: Min scalar for visual normalization (or None to skip normalization)
        s_min_t: Min scalar for textual normalization (or None to skip normalization)
        lambda_val: Harris penalty scalar
        k: Number of expansion neighbors
        gamma: Softmax temperature scale
        return_stats: If True, return diagnostics dict in addition to expanded query
        eps: Numerical stability epsilon

    Returns:
        q_v_expanded: Expanded centered visual query, shape (d,)
        (optional) stats dict
    """
    q_v = q_v.reshape(-1)
    q_t = q_t.reshape(-1)
    if X_v.ndim != 2:
        raise ValueError("X_v must be 2D tensor (N, d)")
    if P.ndim != 2:
        raise ValueError("P must be 2D tensor (d, c)")
    if X_v.size(1) != q_v.numel() or q_t.numel() != q_v.numel() or P.size(0) != q_v.numel():
        raise ValueError("Feature dimensions mismatch among q_v, q_t, X_v, and P")

    # Step 1: preliminary bimodal scoring (fully vectorized over database)
    q_v_proj = P @ (P.T @ q_v)
    s_v = X_v @ q_v_proj
    s_t = X_v @ q_t

    if s_min_v is not None and s_min_t is not None:
        s_min_v_t = torch.as_tensor(s_min_v, device=q_v.device, dtype=q_v.dtype)
        s_min_t_t = torch.as_tensor(s_min_t, device=q_v.device, dtype=q_v.dtype)
        denom_v = torch.abs(s_min_v_t).clamp_min(eps)
        denom_t = torch.abs(s_min_t_t).clamp_min(eps)
        s_v_work = (s_v - s_min_v_t) / denom_v
        s_t_work = (s_t - s_min_t_t) / denom_t
    else:
        s_v_work = s_v
        s_t_work = s_t

    # Rectification to reduce drift from negative preliminary matches
    s_v_work = torch.clamp(s_v_work, min=0)
    s_t_work = torch.clamp(s_t_work, min=0)
    s_f = s_v_work * s_t_work - lambda_val * (s_v_work + s_t_work) ** 2

    # Step 2: top-k selection and anchoring
    n_db = X_v.size(0)
    k_eff = min(max(int(k), 1), n_db)
    top_scores, top_indices = torch.topk(s_f, k_eff, largest=True, sorted=True)
    top_features = X_v[top_indices]

    anchor_score = top_scores.max()
    z = torch.cat([top_features, q_v.reshape(1, -1)], dim=0)
    fused_scores = torch.cat([top_scores, anchor_score.reshape(1)], dim=0)

    # Step 3: bimodal weighting and expansion
    weights = F.softmax(gamma * fused_scores, dim=0)
    q_v_expanded = (weights.unsqueeze(1) * z).sum(dim=0)

    if not return_stats:
        return q_v_expanded

    stats = {
        "topk_mean_fusion": float(top_scores.mean().item()),
        "anchor_weight": float(weights[-1].item()),
        "weight_entropy": float((-weights * torch.log(weights.clamp_min(eps))).sum().item()),
        "k_eff": int(k_eff),
    }
    return q_v_expanded, stats


class DynamicSubspaceProjector:
    """
    Query-Adaptive Dynamic Subspace Projection (QASP).

    Precomputes static positive covariance and stores centered negative corpus features.
    At query time, constructs query-conditioned dynamic projection via:
      1) cosine weighting against negative concepts
      2) weighted dynamic negative covariance
      3) eigendecomposition of dynamic covariance
    """

    def __init__(self, x_plus_centered, x_minus_centered, eps=1e-8):
        if x_plus_centered.ndim != 2 or x_minus_centered.ndim != 2:
            raise ValueError("x_plus_centered and x_minus_centered must be 2D tensors")
        if x_plus_centered.shape[1] != x_minus_centered.shape[1]:
            raise ValueError("x_plus_centered and x_minus_centered must have same feature dimension")

        self.eps = float(eps)
        self.x_plus = x_plus_centered
        self.x_minus = x_minus_centered
        self.device = x_minus_centered.device
        self.dtype = x_minus_centered.dtype
        self.num_neg, self.dim = x_minus_centered.shape

        if self.num_neg <= 0:
            raise ValueError("x_minus_centered must contain at least one negative sample")

        denom = max(x_plus_centered.size(0) - 1, 1)
        self.c_plus = (x_plus_centered.T @ x_plus_centered) / float(denom)
        self.identity = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        self.uniform_w = torch.full(
            (self.num_neg,), 1.0 / float(self.num_neg), device=self.device, dtype=self.dtype
        )

    def _compute_weights(self, q_t, beta):
        if beta < 1.0:
            raise ValueError(f"beta must be >= 1.0, got {beta}")

        q_t = q_t.reshape(-1)
        if q_t.numel() != self.dim:
            raise ValueError(f"q_t has dim {q_t.numel()}, expected {self.dim}")

        q_norm = q_t.norm().clamp_min(self.eps)
        x_norm = self.x_minus.norm(dim=1).clamp_min(self.eps)
        sims = (self.x_minus @ q_t) / (x_norm * q_norm)

        weights_unnorm = torch.relu(sims).pow(beta)
        weight_sum = weights_unnorm.sum()
        used_uniform = bool(weight_sum <= self.eps)
        if used_uniform:
            weights = self.uniform_w
        else:
            weights = weights_unnorm / weight_sum

        return sims, weights, used_uniform

    def get_dynamic_projection_matrix(self, q_t, beta, alpha, k):
        """
        Build a query-adaptive projection basis and projector matrix.

        Args:
            q_t: Centered text query embedding (d,)
            beta: ReLU exponent (>=1)
            alpha: Negative covariance weight
            k: Number of principal components to keep

        Returns:
            Dictionary with:
              - basis: (d, k_eff) principal directions
              - projector: (d, d) projection matrix
              - num_positive_eigs, active_negatives, weight_entropy, max_weight, min_weight
              - used_uniform_weights, used_identity_fallback
        """
        k = int(k)
        sims, weights, used_uniform = self._compute_weights(q_t=q_t, beta=beta)

        # Vectorized dynamic covariance: (X_- * w)^T @ X_-
        c_minus_dynamic = (self.x_minus * weights.unsqueeze(1)).T @ self.x_minus
        c_dynamic = (1.0 - alpha) * self.c_plus - alpha * c_minus_dynamic
        c_dynamic = 0.5 * (c_dynamic + c_dynamic.T)

        evals, evecs = torch.linalg.eigh(c_dynamic)
        positive_mask = evals > 0
        num_positive = int(positive_mask.sum().item())

        used_identity_fallback = False
        if k <= 0 or num_positive == 0:
            basis = self.identity[:, :0]
            projector = self.identity
            used_identity_fallback = True
        else:
            evals_pos = evals[positive_mask]
            evecs_pos = evecs[:, positive_mask]
            k_eff = min(k, evecs_pos.size(1))
            if k_eff <= 0:
                basis = self.identity[:, :0]
                projector = self.identity
                used_identity_fallback = True
            else:
                basis = torch.flip(evecs_pos[:, -k_eff:], dims=[1])
                projector = basis @ basis.T

        active_negatives = int((weights > 0).sum().item())
        weight_entropy = float((-weights * torch.log(weights.clamp_min(self.eps))).sum().item())
        max_weight = float(weights.max().item())
        min_weight = float(weights.min().item())

        return {
            "basis": basis,
            "projector": projector,
            "num_positive_eigs": num_positive,
            "active_negatives": active_negatives,
            "weight_entropy": weight_entropy,
            "max_weight": max_weight,
            "min_weight": min_weight,
            "used_uniform_weights": used_uniform,
            "used_identity_fallback": used_identity_fallback,
        }

    def project_batch(self, image_queries, text_queries, beta, alpha, k):
        """
        Project a batch of image queries using query-conditioned projectors.
        Uses per-query eigendecomposition; dynamic covariance computation is vectorized.
        """
        if image_queries.ndim != 2 or text_queries.ndim != 2:
            raise ValueError("image_queries and text_queries must be 2D tensors")
        if image_queries.shape != text_queries.shape:
            raise ValueError("image_queries and text_queries must have the same shape")

        projected = []
        active_negatives = []
        weight_entropy = []
        max_weight = []
        min_weight = []
        num_positive_eigs = []

        for q_idx in range(text_queries.size(0)):
            dynamic = self.get_dynamic_projection_matrix(
                q_t=text_queries[q_idx],
                beta=beta,
                alpha=alpha,
                k=k,
            )
            projected_q = image_queries[q_idx] @ dynamic["projector"]
            projected.append(projected_q)

            active_negatives.append(dynamic["active_negatives"])
            weight_entropy.append(dynamic["weight_entropy"])
            max_weight.append(dynamic["max_weight"])
            min_weight.append(dynamic["min_weight"])
            num_positive_eigs.append(dynamic["num_positive_eigs"])

        projected = torch.stack(projected, dim=0) if projected else image_queries
        stats = {
            "active_negatives": torch.tensor(active_negatives, device=self.device, dtype=torch.int64),
            "weight_entropy": torch.tensor(weight_entropy, device=self.device, dtype=self.dtype),
            "max_weight": torch.tensor(max_weight, device=self.device, dtype=self.dtype),
            "min_weight": torch.tensor(min_weight, device=self.device, dtype=self.dtype),
            "num_positive_eigs": torch.tensor(num_positive_eigs, device=self.device, dtype=torch.int64),
        }
        return projected, stats


def calculate_rankings(args, image_features, text_features, database_features, return_aux=False):
    """
    Calculate retrieval rankings using specified method.
    
    Args:
        args: Configuration with method, backbone, and algorithm parameters
        image_features: Query image features (num_queries, dim)
        text_features: Query text features (num_queries, dim)
        database_features: Database image features (num_database, dim)
    
    Returns:
        rankings: Tensor of ranked database indices (num_queries, num_database)
        aux (optional): Extra method-specific outputs when return_aux=True
    """
    device = image_features.device
    method_name = args.method.lower()
    aux = {}
    
    # Features are pre-normalized, but ensure normalization for safety
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    database_features = database_features / database_features.norm(dim=1, keepdim=True)


    # ===== Simple Baseline Methods =====
    if method_name == "sum":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        total_sim = sim_img + sim_text
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif method_name == "text":
        total_sim = text_features @ database_features.t()
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif method_name == "image":
        total_sim = image_features @ database_features.t()
        ranks = torch.argsort(total_sim.cpu(), descending=True)
        
    elif method_name == "product":
        sim_img = image_features @ database_features.t()
        sim_text = text_features @ database_features.t()
        # Rectify similarities (remove negative values)
        sim_img = torch.clamp(sim_img, min=0)
        sim_text = torch.clamp(sim_text, min=0)
        total_sim = sim_img * sim_text
        ranks = torch.argsort(total_sim.cpu(), descending=True)
    
    # ===== Proposed Method (Basic) =====
    # note that this implementation is not as efficient as possible. See paper for details.
    elif method_name in ("basic", "mahf", "qasp", "tgbqe"):

        # Load text corpora for BASIC method
        corpus_dir = os.path.join("features", f"{args.backbone}_features", "corpus")
        
        # Positive corpus (objects)
        pos_corpus_file = os.path.join(corpus_dir, f"{args.specified_corpus}.pkl")
        text_corpus_pos, _ = read_corpus(pos_corpus_file, device, norm=args.norm)
        text_corpus_pos = text_corpus_pos / text_corpus_pos.norm(dim=1, keepdim=True)
        
        # Negative corpus (styles)
        neg_corpus_file = os.path.join(corpus_dir, f"{args.specified_ncorpus}.pkl")
        text_corpus_neg, _ = read_corpus(neg_corpus_file, device, norm=args.norm)
        text_corpus_neg = text_corpus_neg / text_corpus_neg.norm(dim=1, keepdim=True)



        if args.standardize_features:
            
            mean_img = database_features.mean(0, keepdim=True)
            mean_txt = text_corpus_pos.mean(0, keepdim=True)

            if args.use_laion_mean:
                if args.backbone == "clip":
                    with open('./data/laion_mean/laion_1m_mean_clip.pkl', mode='rb') as f:
                        data = pickle.load(f)
                        mean_img = data['laion_1m_mean'].to(device)
                elif args.backbone == "siglip":
                    with open('./data/laion_mean/laion_1m_mean_siglip.pkl', mode='rb') as f:
                        data = pickle.load(f)
                        mean_img = data['laion_1m_mean'].to(device)

            centered_database_features = database_features - mean_img
            centered_image_features = image_features - mean_img

            centered_corpus_pos_features = text_corpus_pos - mean_txt
            centered_corpus_neg_features = text_corpus_neg - mean_txt

            centered_text_features = text_features - mean_txt 
        else:
            centered_database_features, centered_image_features = database_features, image_features
            centered_corpus_pos_features, centered_corpus_neg_features = text_corpus_pos, text_corpus_neg
            centered_text_features = text_features

        if method_name == "mahf":
            alignment, alignment_01 = _normalized_cosine_alignment(
                centered_image_features, centered_text_features
            )
            lambda_q = _adaptive_harris_lambda(args, alignment_01)
            aux = {
                "alignment": alignment.detach().cpu(),
                "alignment_01": alignment_01.detach().cpu(),
                "lambda_q": lambda_q.detach().cpu(),
            }
            
        projection_matrix = torch.eye(centered_database_features.shape[1], device=device)
        projection_basis = projection_matrix
        if args.project_features:
            aa = args.aa
            A, B = centered_corpus_pos_features, centered_corpus_neg_features

            Nc = int(args.num_principal_components_for_projection)
            if method_name == "qasp":
                qasp_projector = DynamicSubspaceProjector(
                    x_plus_centered=A,
                    x_minus_centered=B,
                )
                proj_image_features, qasp_stats = qasp_projector.project_batch(
                    image_queries=centered_image_features,
                    text_queries=centered_text_features,
                    beta=float(args.qasp_beta),
                    alpha=float(aa),
                    k=Nc,
                )
                aux.update({
                    "qasp_active_negatives": qasp_stats["active_negatives"].detach().cpu(),
                    "qasp_weight_entropy": qasp_stats["weight_entropy"].detach().cpu(),
                    "qasp_max_weight": qasp_stats["max_weight"].detach().cpu(),
                    "qasp_min_weight": qasp_stats["min_weight"].detach().cpu(),
                    "qasp_num_positive_eigs": qasp_stats["num_positive_eigs"].detach().cpu(),
                })
            else:
                # Compute scatter matrices
                Sa = A.T @ A / (A.size(0) - 1)
                Sb = B.T @ B / (B.size(0) - 1)

                C = (1 - aa) * (Sa) - aa * Sb + 1e-5
                L, Vy_t2 = torch.linalg.eigh(C)
                L = L.flip(dims=[0])
                Vy_t = -Vy_t2.flip(dims=[1]).T

                mask_l = L > 0
                Nc = min(Nc, sum(mask_l).item())
                Vy_t = Vy_t[:Nc]

                projection_basis = Vy_t.T
                projection_matrix = Vy_t.T @ Vy_t
                proj_image_features = centered_image_features @ projection_matrix
        else:
            proj_image_features = centered_image_features

        proj_database_features = centered_database_features 

        sim_img_min, sim_text_min = None, None
        if args.normalize_similarities:
            file_path = os.path.join(args.path_to_synthetic_data, f"dataset_1_sd_{args.backbone}.pkl.npy")
            generated_dataset = np.load(file_path, allow_pickle=True).item()
            image_features_generated = torch.Tensor(generated_dataset['image_features']).to(device)
            text_features_generated = torch.Tensor(generated_dataset['text_features']).to(device)

            image_features_generated = image_features_generated / image_features_generated.norm(dim=1, keepdim=True)
            text_features_generated = text_features_generated / text_features_generated.norm(dim=1, keepdim=True)

            if args.standardize_features:
                image_features_generated -= mean_img
                text_features_generated -= mean_txt

            sim_img_gen = image_features_generated @ image_features_generated.t()
            sim_text_gen = text_features_generated @ image_features_generated.t()
            sim_img_min = sim_img_gen.cpu().min()
            sim_text_min = sim_text_gen.cpu().min()


        # query expansion
        if args.do_query_expansion:
            if method_name == "tgbqe":
                expanded_queries = []
                topk_mean_fusion = []
                anchor_weight = []
                weight_entropy = []

                for q_idx in range(centered_image_features.size(0)):
                    q_expanded, q_stats = apply_tg_bqe(
                        q_v=centered_image_features[q_idx],
                        q_t=centered_text_features[q_idx],
                        X_v=centered_database_features,
                        P=projection_basis,
                        s_min_v=sim_img_min if args.normalize_similarities else None,
                        s_min_t=sim_text_min if args.normalize_similarities else None,
                        lambda_val=float(args.harris_lambda),
                        k=int(args.tgbqe_k),
                        gamma=float(args.tgbqe_gamma),
                        return_stats=True,
                    )
                    expanded_queries.append(q_expanded)
                    topk_mean_fusion.append(q_stats["topk_mean_fusion"])
                    anchor_weight.append(q_stats["anchor_weight"])
                    weight_entropy.append(q_stats["weight_entropy"])

                expanded_queries = torch.stack(expanded_queries, dim=0)
                proj_image_features = expanded_queries @ projection_matrix
                aux.update({
                    "tgbqe_topk_mean_fusion": torch.tensor(topk_mean_fusion, device=device, dtype=centered_image_features.dtype).cpu(),
                    "tgbqe_anchor_weight": torch.tensor(anchor_weight, device=device, dtype=centered_image_features.dtype).cpu(),
                    "tgbqe_weight_entropy": torch.tensor(weight_entropy, device=device, dtype=centered_image_features.dtype).cpu(),
                })
            else:
                extra_features = centered_database_features
                init_sim_img = proj_image_features @ extra_features.t()

                top_k = min(25, extra_features.shape[0])
                top_values, top_indices = torch.topk(init_sim_img, top_k)
                top_features = (centered_database_features).cpu()[top_indices.cpu()]

                # add original features
                top_features = torch.cat((top_features, centered_image_features.unsqueeze(1).cpu()), dim=1)
                # add 1 similarity for original features
                top_values = torch.cat((top_values, torch.ones((top_values.shape[0], 1)).to(device)), dim=1)

                # top values as exponential over cosine similarity
                top_values = torch.exp(.1 * top_values)
                top_values = top_values / top_values.sum(dim=1).unsqueeze(-1)

                # weighted mean
                top_features = top_features * top_values.unsqueeze(-1).cpu()
                top_features = top_features.sum(dim=1).to(device)
                if method_name == "qasp":
                    qasp_projector = DynamicSubspaceProjector(
                        x_plus_centered=centered_corpus_pos_features,
                        x_minus_centered=centered_corpus_neg_features,
                    )
                    proj_image_features, _ = qasp_projector.project_batch(
                        image_queries=top_features,
                        text_queries=centered_text_features,
                        beta=float(args.qasp_beta),
                        alpha=float(args.aa),
                        k=int(args.num_principal_components_for_projection),
                    )
                else:
                    proj_image_features = (top_features) @ projection_matrix



        sim_img = (proj_image_features) @ (proj_database_features).t()
        sim_img = sim_img.cpu()

        sim_text = centered_text_features @ (centered_database_features).t()
        sim_text = sim_text.cpu()


        if args.normalize_similarities:
            sim_text = (sim_text - sim_text_min)/sim_img_min.abs()        
            sim_img = (sim_img - sim_img_min)/sim_img_min.abs()

        # Rectify similarities (remove negative values)
        sim_text = torch.clamp(sim_text, min=0)
        sim_img = torch.clamp(sim_img, min=0)

        # apply harris criterion
        if method_name == "mahf":
            lambda_q_cpu = lambda_q.unsqueeze(1).cpu()
            sim_all = sim_text * sim_img - lambda_q_cpu * (sim_text + sim_img)**2
        else:
            sim_all = sim_text * sim_img - args.harris_lambda * (sim_text + sim_img)**2

        ranks = torch.argsort(sim_all, descending=True)

    if return_aux:
        return ranks, aux

    return ranks

def metrics_calc(
    rankings,
    target_domain,
    current_query_classes,
    database_classes,
    database_domains,
    at,
    mode="composed"  # one of: "composed", "image", "text"
):
    metrics = {}

    class_id_map = {class_name: idx for idx, class_name in enumerate(database_classes)}
    domain_id_map = {domain_name: idx for idx, domain_name in enumerate(database_domains)}

    database_classes_ids = [class_id_map[class_name] for class_name in database_classes]
    database_domains_ids = [domain_id_map[domain_name] for domain_name in database_domains]
    query_classes_ids = [class_id_map[class_name] for class_name in current_query_classes]
    target_domain_id = domain_id_map[target_domain]

    device = rankings.device
    database_classes_tensor = torch.tensor(database_classes_ids).to(device)
    database_domains_tensor = torch.tensor(database_domains_ids).to(device)
    query_classes_tensor = torch.tensor(query_classes_ids).to(device)
    target_domain_tensor = torch.tensor(target_domain_id).to(device)

    # Shape: (num_queries, num_database)
    match_class = (database_classes_tensor[rankings] == query_classes_tensor.unsqueeze(1)).float()
    match_domain = (database_domains_tensor[rankings] == target_domain_tensor).float()

    if mode == "image":
        correct = match_class
    elif mode == "text":
        correct = match_domain
    else:
        correct = match_class * match_domain  # composed case

    metrics["mAP"], AP_list = compute_map(correct.cpu().numpy())

    for k in at:
        correct_k = correct[:, :k]
        num_correct = torch.sum(correct_k, dim=1)
        num_predicted = torch.sum(torch.ones_like(correct_k), dim=1)
        num_total = torch.sum(correct, dim=1)

        recall = torch.mean(num_correct / (num_total + 1e-5))
        precision = torch.mean(num_correct / (torch.minimum(num_total, num_predicted) + 1e-5))

        metrics[f"R@{k}"] = round(recall.item() * 100, 2)
        metrics[f"P@{k}"] = round(precision.item() * 100, 2)

    print(metrics)
    return metrics, AP_list

def map_calc_icir(rankings, db_paths, q_paths, q_instances, q_texts, db_instances, db_texts):
    """
    Calculate mean Average Precision for icir dataset.
    
    A match is correct if both the instance AND text query match between query and database item.
    
    Args:
        rankings: Tensor of shape (num_queries, num_database) with ranked database indices
        db_paths: List of database image paths
        q_paths: List of query image paths
        q_instances: List of query instance identifiers
        q_texts: List of query text descriptions
        db_instances: List of database instance identifiers
        db_texts: List of database text descriptions
    
    Returns:
        Dictionary with:
            - "mAP": Mean average precision (float)
            - "APs": List of per-query average precisions
            - "correct_matrix": Binary tensor indicating correct matches (num_queries, num_database)
    """
    num_queries = rankings.shape[0]
    num_database = rankings.shape[1]
    
    # Create binary correctness matrix
    correct_matrix = torch.zeros_like(rankings, dtype=torch.float32)
    
    # Mark correct matches
    for q_idx in range(num_queries):
        query_instance = q_instances[q_idx]
        query_text = q_texts[q_idx]
        
        # Check each ranked database item
        for rank_pos, db_idx in enumerate(rankings[q_idx].tolist()):
            db_instance = db_instances[db_idx]
            db_text = db_texts[db_idx]
            
            # Match requires both instance AND text to match
            if query_instance == db_instance and query_text == db_text:
                correct_matrix[q_idx, rank_pos] = 1.0
    
    # Compute average precision for each query
    mAP, AP_list = compute_map(correct_matrix.cpu().numpy())
    
    return {
        "mAP": mAP,
        "APs": AP_list,
        "correct_matrix": correct_matrix
    }
