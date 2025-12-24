# evaluate.py
import os
import sys
import time
import json
import math
import yaml
import torch
import clip
import lpips
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mediapipe as mp
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Optional external libs (clean-fid / torch-fidelity / diffsim)
_has_cleanfid = False
_has_torch_fidelity = False
_has_difsim = False
try:
    import cleanfid
    _has_cleanfid = True
except Exception:
    try:
        # try legacy import name
        from cleanfid import fid as cleanfid_fid
        _has_cleanfid = True
    except Exception:
        _has_cleanfid = False

try:
    from torch_fidelity import calculate_metrics
    _has_torch_fidelity = True
except Exception:
    _has_torch_fidelity = False

try:
    import diffsim  # optional, only if user installed DiffSim implementation
    _has_difsim = True
except Exception:
    _has_difsim = False

# -------------------------------------------------------------------------
# 1. Pose Consistency
# -------------------------------------------------------------------------
class PoseConsistencyEvaluator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def extract_keypoints(self, image_pil):
        image_np = np.array(image_pil.convert('RGB'))
        results = self.pose.process(image_np)
        if not results.pose_landmarks:
            return None
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.visibility])
        return np.array(keypoints)

    def calculate_pose_metrics(self, original_img, generated_img):
        kps1 = self.extract_keypoints(original_img)
        kps2 = self.extract_keypoints(generated_img)
        if kps1 is None or kps2 is None:
            return {'pose_mse': np.nan, 'pose_similarity': 0.0}
        threshold = 0.5
        mask = (kps1[:, 2] > threshold) & (kps2[:, 2] > threshold)
        if np.sum(mask) < 5:
            return {'pose_mse': np.nan, 'pose_similarity': 0.0}
        valid_kps1 = kps1[mask, :2]
        valid_kps2 = kps2[mask, :2]
        mse = np.mean(np.sum((valid_kps1 - valid_kps2) ** 2, axis=1))
        distances = np.linalg.norm(valid_kps1 - valid_kps2, axis=1)
        similarity = 1.0 / (1.0 + np.mean(distances) * 10)
        return {'pose_mse': float(mse), 'pose_similarity': float(similarity)}

# -------------------------------------------------------------------------
# 2. Style & Quality Evaluator (扩展)
# -------------------------------------------------------------------------
class StyleQualityEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading models on {self.device}...")
        # LPIPS
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        # Guofeng keywords retained
        self.guofeng_keywords = [
            "Chinese traditional painting style",
            "ink wash painting",
            "elegant oriental aesthetic",
            "wuxia style"
        ]
        # Anime vs photorealistic prompts for style classification (CLIP zero-shot)
        self.anime_prompts = ["anime style", "cartoon", "manga illustration"]
        self.photo_prompts = ["photorealistic photo", "photograph", "realistic photo"]

    # ----- Per-pair metrics -----
    def calculate_lpips(self, img1_pil, img2_pil):
        try:
            # lpips package expects tensors; use helper functions
            tensor1 = lpips.im2tensor(lpips.load_image(img1_pil)).to(self.device)
            tensor2 = lpips.im2tensor(lpips.load_image(img2_pil)).to(self.device)
            with torch.no_grad():
                dist = self.lpips_model(tensor1, tensor2)
            return float(dist.item())
        except Exception as e:
            print(f"[LPIPS] calculation failed: {e}")
            return np.nan

    def calculate_clip_score(self, image_pil, prompt):
        try:
            text_token = clip.tokenize([prompt[:77]]).to(self.device)
            image_tensor = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                text_features = self.clip_model.encode_text(text_token)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).item()
            return float(similarity)
        except Exception as e:
            print(f"[CLIP] calculation failed: {e}")
            return np.nan

    def calculate_style_fidelity(self, image_pil):
        scores = []
        for kw in self.guofeng_keywords:
            sc = self.calculate_clip_score(image_pil, kw)
            if not math.isnan(sc):
                scores.append(sc)
        return float(np.mean(scores)) if scores else np.nan

    def calculate_anime_prob(self, image_pil):
        """
        使用 CLIP zero-shot 将图像映射到「anime vs photo」两类上，
        返回 [anime_prob (0-1)] 。这是一个轻量级样式分类器替代方案。
        """
        try:
            # Compute mean similarity for each class
            anime_scores = [self.calculate_clip_score(image_pil, p) for p in self.anime_prompts]
            photo_scores = [self.calculate_clip_score(image_pil, p) for p in self.photo_prompts]
            anime_mean = np.nanmean(anime_scores) if len(anime_scores) > 0 else np.nan
            photo_mean = np.nanmean(photo_scores) if len(photo_scores) > 0 else np.nan
            if math.isnan(anime_mean) or math.isnan(photo_mean):
                return np.nan
            # convert to probability via softmax
            exps = np.exp([anime_mean, photo_mean])
            probs = exps / np.sum(exps)
            return float(probs[0])
        except Exception as e:
            print(f"[StyleClassifier] failed: {e}")
            return np.nan

    # ----- Set-level metrics (FID/IS/KID) -----
    def compute_set_fidelity(self, gen_dir, ref_dir):
        """
        Try available libraries to compute FID/IS/KID:
        - prefer torch_fidelity.calculate_metrics
        - else try clean-fid
        Returns dict with keys {'fid','is_mean','is_std','kid_mean'} where present.
        """
        metrics = {}
        if _has_torch_fidelity:
            try:
                # torch_fidelity expects folders or lists; use folders
                m = calculate_metrics(input1=gen_dir, input2=ref_dir, cuda=(self.device.startswith('cuda')))
                # keys: 'frechet_inception_distance' etc.
                if 'frechet_inception_distance' in m:
                    metrics['fid'] = float(m['frechet_inception_distance'])
                if 'inception_score_mean' in m:
                    metrics['is_mean'] = float(m['inception_score_mean'])
                    metrics['is_std'] = float(m.get('inception_score_std', 0.0))
                if 'kid_mean' in m:
                    metrics['kid_mean'] = float(m['kid_mean'])
                return metrics
            except Exception as e:
                print(f"[torch-fidelity] failed: {e}")
                # fallthrough to clean-fid
        if _has_cleanfid:
            try:
                # cleanfid API: cleanfid.compute_fid(gen_dir, ref_dir)
                try:
                    fid_val = cleanfid.compute_fid(gen_dir, ref_dir)
                except Exception:
                    # some versions expose function under cleanfid.fid or cleanfid.fid.compute_fid
                    fid_val = cleanfid.fid.compute_fid(gen_dir, ref_dir)
                metrics['fid'] = float(fid_val)
            except Exception as e:
                print(f"[clean-fid] failed: {e}")
        if not metrics:
            print("[SetFidelity] Neither torch-fidelity nor clean-fid available; skipping set-level fidelity metrics.")
        return metrics

    # ----- DiffSim placeholder -----
    def compute_difsim_scores(self, gen_dir, ref_dir):
        """If diffsim available, compute its similarity metrics. Placeholder/wrapper."""
        if not _has_difsim:
            print("[DiffSim] not installed; skipping.")
            return {}
        try:
            # This requires diffsim API — keep generic wrapper to adapt to user's diffsim installation
            # Example: diffsim.evaluate_pairwise(gen_dir, ref_dir)
            res = diffsim.evaluate(gen_dir, ref_dir)  # user must provide their diffsim API
            return res
        except Exception as e:
            print(f"[DiffSim] evaluation failed: {e}")
            return {}

# -------------------------------------------------------------------------
# 3. AestheticEvaluator (unchanged placeholder)
# -------------------------------------------------------------------------
class AestheticEvaluator:
    def __init__(self, device='cpu'):
        self.model_loaded = False
    def predict_score(self, image_pil):
        if not self.model_loaded:
            return -1.0
        return 5.0

# -------------------------------------------------------------------------
# 4. EvaluationPipeline（扩展 baseline / set metrics / diffsim / anime_prob）
# -------------------------------------------------------------------------
class EvaluationPipeline:
    def __init__(self, exp_dir, output_dir="eval", fid_ref_dir=None, baseline_dir=None, run_difsim=False):
        self.exp_dir = Path(exp_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.exp_dir / "metadata.csv"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        self.pose_eval = PoseConsistencyEvaluator()
        self.style_eval = StyleQualityEvaluator()
        self.aes_eval = AestheticEvaluator()
        self.fid_ref_dir = Path(fid_ref_dir) if fid_ref_dir else None
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.run_difsim = run_difsim

    def run_evaluation(self):
        print(f"Starting evaluation for: {self.exp_dir.name}")
        df = pd.read_csv(self.metadata_path)
        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            orig_path = self.exp_dir / row['original_image_path']
            gen_path = self.exp_dir / row['generated_image_path']

            if not orig_path.exists() or not gen_path.exists():
                print(f"Warning: Image missing at index {idx} ({orig_path} or {gen_path})")
                continue

            try:
                orig_img = Image.open(orig_path).convert('RGB')
                gen_img = Image.open(gen_path).convert('RGB')
            except Exception as e:
                print(f"Error opening image: {e}")
                continue

            res_entry = {
                'filename': gen_path.name,
                'original_image': row['original_image_path'],
                'prompt': row.get('prompt', ''),
                'seed': row.get('seed', 0),
                'inference_time': row.get('inference_time', np.nan)
            }

            # Pose
            pose_metrics = self.pose_eval.calculate_pose_metrics(orig_img, gen_img)
            res_entry.update(pose_metrics)

            # LPIPS
            res_entry['lpips'] = self.style_eval.calculate_lpips(orig_img, gen_img)

            # CLIP score for prompt (if provided)
            if 'prompt' in row and isinstance(row['prompt'], str) and row['prompt'].strip():
                res_entry['clip_score'] = self.style_eval.calculate_clip_score(gen_img, row['prompt'])
            else:
                res_entry['clip_score'] = np.nan

            # Style fidelity (guofeng)
            res_entry['style_fidelity'] = self.style_eval.calculate_style_fidelity(gen_img)

            # Anime probability (CLIP zero-shot style classifier)
            res_entry['anime_prob'] = self.style_eval.calculate_anime_prob(gen_img)

            results.append(res_entry)

        self.results_df = pd.DataFrame(results)
        csv_path = self.output_dir / f"results_{self.exp_dir.name}.csv"
        self.results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")

        # Compute set-level metrics (FID/IS/KID) if reference dir provided
        set_metrics = {}
        if self.fid_ref_dir and self.fid_ref_dir.exists():
            # prepare a folder with generated images (from results_df)
            gen_images_dir = self.output_dir / f"gen_images_{self.exp_dir.name}"
            gen_images_dir.mkdir(exist_ok=True)
            # copy or save generated images paths
            for i, row in self.results_df.iterrows():
                src = self.exp_dir / row['original_image']  # note: original_image field currently points to original path; user's metadata might list generated differently
                # prefer to use stored filename of generated: find actual generated path from metadata if present
                # We'll search by filename in exp_dir generated paths
                candidate = list(self.exp_dir.glob(f"**/{row['filename']}"))
                if candidate:
                    try:
                        dst_path = gen_images_dir / row['filename']
                        if not dst_path.exists():
                            from shutil import copyfile
                            copyfile(candidate[0], dst_path)
                    except Exception:
                        pass

            # only compute if there are images
            if any(gen_images_dir.iterdir()):
                set_metrics = self.style_eval.compute_set_fidelity(str(gen_images_dir), str(self.fid_ref_dir))
                # store set_metrics
                for k, v in set_metrics.items():
                    # attach to pipeline-level attribute
                    setattr(self, f"set_{k}", v)

        # Baseline comparison: compute same set metrics on baseline_dir if provided
        baseline_metrics = {}
        if self.baseline_dir and self.baseline_dir.exists():
            # If baseline is a folder of images (matching generated set), compute set-level metrics
            if self.fid_ref_dir and self.fid_ref_dir.exists():
                baseline_metrics = self.style_eval.compute_set_fidelity(str(self.baseline_dir), str(self.fid_ref_dir))
            # If baseline images correspond file-by-file to our results_df, optionally compute average LPIPS/CLIP/anime_prob: skipped here (could implement if mapping known)

        # Optionally run diffsim
        diffsim_metrics = {}
        if self.run_difsim and _has_difsim:
            # attempt to evaluate using diffsim; user must ensure diffsim usage matches their environment
            diffsim_metrics = self.style_eval.compute_difsim_scores(str(gen_images_dir), str(self.fid_ref_dir)) if ('gen_images_dir' in locals() and self.fid_ref_dir) else {}
        elif self.run_difsim:
            print("[DiffSim] requested but diffsim not installed; skipping.")

        # Attach set-level & baseline metrics to pipeline for reporting
        self.set_metrics = set_metrics
        self.baseline_metrics = baseline_metrics
        self.difsim_metrics = diffsim_metrics

        return self.results_df

    def generate_report(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to report.")
            return
        df = self.results_df

        numeric_cols = ['pose_mse', 'lpips', 'clip_score', 'style_fidelity', 'anime_prob', 'inference_time']
        summary_stats = df[numeric_cols].mean(skipna=True).to_dict()
        summary_std = df[numeric_cols].std(skipna=True).to_dict()

        # Grouped stats
        grouped = df.groupby('original_image')
        group_stats = pd.DataFrame({
            'mean_clip': grouped['clip_score'].mean(),
            'max_clip': grouped['clip_score'].max(),
            'mean_pose_mse': grouped['pose_mse'].mean(),
            'min_pose_mse': grouped['pose_mse'].min(),
            'count': grouped.size()
        })

        # Visualizations
        self._plot_boxplots(df, numeric_cols)
        self._plot_correlation(df, numeric_cols)
        self._plot_radar_chart(summary_stats)

        # Compare set-level metrics with baseline if available
        comparison = {}
        if getattr(self, 'set_metrics', None):
            for k, v in self.set_metrics.items():
                current = v
                baseline = self.baseline_metrics.get(k) if getattr(self, 'baseline_metrics', None) else None
                delta = None
                pct = None
                if baseline is not None:
                    # For metrics where lower is better (fid, kid_mean, lpips, pose_mse), compute improvement as (baseline - current)/baseline
                    if k in ['fid', 'kid_mean']:
                        delta = baseline - current
                        pct = (delta / baseline * 100.0) if baseline != 0 else None
                    else:
                        delta = current - baseline
                        pct = (delta / baseline * 100.0) if baseline != 0 else None
                comparison[k] = {'current': current, 'baseline': baseline, 'delta': delta, 'pct': pct}

        # Save report markdown
        report_path = self.output_dir / f"summary_{self.exp_dir.name}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Experiment Report: {self.exp_dir.name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write("| Metric | Mean | Std Dev | Interpretation |\n")
            f.write("| :--- | :---: | :---: | :--- |\n")
            f.write(f"| **Pose MSE** | {summary_stats.get('pose_mse', float('nan')):.6f} | {summary_std.get('pose_mse', float('nan')):.6f} | Lower is better |\n")
            f.write(f"| **LPIPS** | {summary_stats.get('lpips', float('nan')):.6f} | {summary_std.get('lpips', float('nan')):.6f} | Lower = closer to source structure |\n")
            f.write(f"| **CLIP Score** | {summary_stats.get('clip_score', float('nan')):.6f} | {summary_std.get('clip_score', float('nan')):.6f} | Higher is better |\n")
            f.write(f"| **Style Fidelity** | {summary_stats.get('style_fidelity', float('nan')):.6f} | {summary_std.get('style_fidelity', float('nan')):.6f} | Higher = more 'Guofeng' style |\n")
            f.write(f"| **Anime Probability** | {summary_stats.get('anime_prob', float('nan')):.6f} | {summary_std.get('anime_prob', float('nan')):.6f} | Higher => more anime-like |\n")
            f.write("\n")

            f.write("## 2. Set-level Metrics (FID/IS/KID) & Baseline Comparison\n\n")
            if getattr(self, 'set_metrics', None):
                f.write("| Metric | Current | Baseline | Delta | % Change |\n")
                f.write("| :--- | :---: | :---: | :---: | :---: |\n")
                for k, v in self.set_metrics.items():
                    baseline_val = self.baseline_metrics.get(k) if getattr(self, 'baseline_metrics', None) else None
                    cur = v
                    if baseline_val is None:
                        f.write(f"| {k} | {cur:.4f} | - | - | - |\n")
                    else:
                        # Interpret direction: for FID smaller is better
                        if k in ['fid', 'kid_mean']:
                            delta = baseline_val - cur
                            pct = (delta / baseline_val * 100.0) if baseline_val != 0 else float('nan')
                            f.write(f"| {k} | {cur:.4f} | {baseline_val:.4f} | {delta:.4f} | {pct:.2f}% |\n")
                        else:
                            delta = cur - baseline_val
                            pct = (delta / baseline_val * 100.0) if baseline_val != 0 else float('nan')
                            f.write(f"| {k} | {cur:.4f} | {baseline_val:.4f} | {delta:.4f} | {pct:.2f}% |\n")
            else:
                f.write("No set-level fidelity metrics were computed (missing `fid_ref_dir`, or required libs not installed).\n")

            f.write("\n## 3. Data Insights\n")
            f.write(f"- Total Images Evaluated: {len(df)}\n")
            f.write(f"- Unique Source Images: {len(group_stats)}\n")
            f.write(f"- Images with failed Pose Detect: {df['pose_mse'].isna().sum()}\n\n")

            f.write("## 4. Visualizations\n")
            f.write(f"![Boxplots](charts/boxplots_{self.exp_dir.name}.png)\n\n")
            f.write(f"![Correlation](charts/correlation_{self.exp_dir.name}.png)\n\n")
            f.write(f"![Radar](charts/radar_{self.exp_dir.name}.png)\n\n")

            f.write("## 5. Best & Worst Examples\n")
            if df['clip_score'].notna().any():
                best_clip = df.loc[df['clip_score'].idxmax()]
                worst_clip = df.loc[df['clip_score'].idxmin()]
                f.write("### Best CLIP Score (Text Alignment)\n")
                f.write(f"- **Score:** {best_clip['clip_score']:.4f}\n")
                f.write(f"- **Prompt:** {best_clip['prompt']}\n")
                f.write(f"- **File:** `{best_clip['filename']}`\n\n")
                f.write("### Worst CLIP Score (Text Alignment)\n")
                f.write(f"- **Score:** {worst_clip['clip_score']:.4f}\n")
                f.write(f"- **File:** `{worst_clip['filename']}`\n\n")
            else:
                f.write("No CLIP score data available.\n\n")

            if df['pose_mse'].notna().any():
                best_pose = df.loc[df['pose_mse'].idxmin()]
                f.write("### Best Pose Consistency (Lowest MSE)\n")
                f.write(f"- **MSE:** {best_pose['pose_mse']:.6f}\n")
                f.write(f"- **File:** `{best_pose['filename']}`\n\n")
            else:
                f.write("No valid pose data detected.\n\n")

            f.write("## 6. Notes on Additional Methods\n")
            if self.run_difsim:
                if self.difsim_metrics:
                    f.write("- DiffSim metrics were computed and attached.\n")
                else:
                    f.write("- DiffSim requested but not computed (library missing or error).\n")
            f.write("- Set-level metrics (FID/IS/KID) require `clean-fid` or `torch-fidelity` packages. If you need reproducible FID for anime domain, prefer a curated anime reference folder passed via `--fid_ref_dir`.\n")
            f.write("- Anime probability is computed by CLIP zero-shot prompts as a lightweight style classifier; for higher-fidelity style classification, consider training a dedicated classifier on anime vs photo data and replacing `calculate_anime_prob`.\n")

        print(f"Report generated at {report_path}")

    # --- Visual helpers (kept similar) ---
    def _plot_boxplots(self, df, cols):
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return
        plt.figure(figsize=(15, 5))
        melted = df[valid_cols].melt(var_name='Metric', value_name='Score')
        sns.boxplot(data=melted, x='Metric', y='Score')
        plt.title("Distribution of Evaluation Metrics")
        plt.savefig(self.chart_dir / f"boxplots_{self.exp_dir.name}.png", dpi=150)
        plt.close()

    def _plot_correlation(self, df, cols):
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) < 2:
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[valid_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Metric Correlation Heatmap")
        plt.savefig(self.chart_dir / f"correlation_{self.exp_dir.name}.png", dpi=150)
        plt.close()

    def _plot_radar_chart(self, summary_stats):
        labels = ['clip_score', 'style_fidelity', 'anime_prob']
        values = [summary_stats.get(l, 0) for l in labels]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25)
        ax.plot(angles, values, linewidth=2)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Average Performance Radar")
        plt.savefig(self.chart_dir / f"radar_{self.exp_dir.name}.png", dpi=150)
        plt.close()

# -------------------------------------------------------------------------
# 5. CLI Entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Painting Enhanced Evaluation Pipeline")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory containing metadata.csv")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save reports")
    parser.add_argument("--fid_ref_dir", type=str, default=None, help="Reference images folder for FID (e.g., real anime set)")
    parser.add_argument("--baseline_dir", type=str, default=None, help="Baseline generated images folder (for set-level metric comparison)")
    parser.add_argument("--run_diffsim", action='store_true', help="Attempt computing DiffSim metrics if diffsim is installed")
    args = parser.parse_args()

    pipeline = EvaluationPipeline(args.exp_dir, args.output_dir, fid_ref_dir=args.fid_ref_dir, baseline_dir=args.baseline_dir, run_difsim=args.run_diffsim)
    pipeline.run_evaluation()
    pipeline.generate_report()
