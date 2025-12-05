import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

# Relative import since both files are in src/
from .rl_datasets import BaseTask, RLDataset, get_task

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

    def tokenize(self, text: str):
        return self.tokenizer(text, add_special_tokens=False)

    def detokenize(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )


class MemoryEfficientAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


@dataclass
class Episode:
    prefix: str
    text: str
    prefix_token_ids: list[int]
    generated_token_ids: list[int]
    reward: float
    reward_info: dict[str, float]
    embeddings: Optional[torch.Tensor] = None
    old_log_probs: Optional[torch.Tensor] = None


def get_or_download_model(model_path_or_id: str) -> str:
    if os.path.exists(model_path_or_id):
        logger.info(f"Model found locally at: {model_path_or_id}")
        return model_path_or_id

    current_dir = Path.cwd()
    models_root = current_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    model_name = model_path_or_id.split("/")[-1]
    local_model_path = models_root / model_name

    if local_model_path.exists() and (local_model_path / "config.json").exists():
        logger.info(f"Model already downloaded at: {local_model_path}")
        return str(local_model_path)

    logger.info(f"Model not found locally. Downloading {model_path_or_id} to {local_model_path}...")
    try:
        snapshot_download(
            repo_id=model_path_or_id, local_dir=local_model_path, local_dir_use_symlinks=False, resume_download=True
        )
        logger.info("Download complete.")
        return str(local_model_path)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise e


def robust_sinkhorn(old_emb, new_emb, mask, window_size=16, eps=0.1, iter=5):
    old_emb, new_emb = old_emb.float(), new_emb.float()
    old_emb = F.normalize(old_emb, p=2, dim=-1)
    new_emb = F.normalize(new_emb, p=2, dim=-1)
    cost = 1.0 - torch.bmm(old_emb, new_emb.transpose(1, 2))
    B, T, _ = old_emb.shape
    idx = torch.arange(T, device=old_emb.device)
    band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= window_size
    valid = band.unsqueeze(0) & (mask.unsqueeze(2) * mask.unsqueeze(1) > 0.5)
    cost_masked = cost.masked_fill(~valid, 1e5)
    log_K = -cost_masked / eps
    u, v = torch.zeros(B, T, device=old_emb.device), torch.zeros(B, T, device=old_emb.device)
    row_mask = mask.bool()
    for _ in range(iter):
        u = -torch.logsumexp(log_K + v.unsqueeze(1), 2).masked_fill(~row_mask, 0.0)
        v = -torch.logsumexp(log_K.transpose(1, 2) + u.unsqueeze(1), 2).masked_fill(~row_mask, 0.0)
    P = torch.exp(u.unsqueeze(2) + v.unsqueeze(1) + log_K) * valid.float()
    return (P * cost).sum((1, 2)) / (mask.sum(1) + 1e-6)


@torch.no_grad()
def rollout(
    model, batch, tokenizer, task: BaseTask, max_gen_len, num_samples, device, loss_type="ot", inference_batch_size=32
):
    total_bsz = len(batch.prefix) * num_samples
    all_prefix_ids = [ids for ids in batch.prefix_token_ids for _ in range(num_samples)]
    all_references = [r for r in batch.references for _ in range(num_samples)]
    all_prefixes = [p for p in batch.prefix for _ in range(num_samples)]

    all_episodes = []

    for start_idx in range(0, total_bsz, inference_batch_size):
        end_idx = min(start_idx + inference_batch_size, total_bsz)
        chunk_bsz = end_idx - start_idx

        chunk_prefix_ids = all_prefix_ids[start_idx:end_idx]
        chunk_prefixes = all_prefixes[start_idx:end_idx]
        chunk_refs = all_references[start_idx:end_idx]

        # Left Padding Logic
        max_p_len = max(len(p) for p in chunk_prefix_ids)
        input_ids = torch.full((chunk_bsz, max_p_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        for i, p in enumerate(chunk_prefix_ids):
            p_len = len(p)
            input_ids[i, -p_len:] = torch.tensor(p, device=device)

        saved_embs = [[] for _ in range(chunk_bsz)]
        saved_lps = [[] for _ in range(chunk_bsz)]
        generated_tokens = [[] for _ in range(chunk_bsz)]
        finished = torch.zeros(chunk_bsz, dtype=torch.bool, device=device)
        past_key_values = None
        curr_input = input_ids

        for step in range(max_gen_len):
            outputs = model(
                input_ids=curr_input, past_key_values=past_key_values, use_cache=True, output_hidden_states=True
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)

            lp = torch.log_softmax(next_logits, dim=-1)
            sel_lp = torch.gather(lp, 1, next_token.unsqueeze(1)).squeeze(1).float().cpu()
            for i in range(chunk_bsz):
                if not finished[i]:
                    saved_lps[i].append(sel_lp[i])

            if loss_type == "ot":
                h = outputs.hidden_states[-1][:, -1, :].float().cpu()
                for i in range(chunk_bsz):
                    if not finished[i]:
                        saved_embs[i].append(h[i])

            finished |= next_token == tokenizer.eos_token_id
            for i in range(chunk_bsz):
                if not finished[i]:
                    generated_tokens[i].append(next_token[i].item())

            if finished.all():
                break
            curr_input = next_token.unsqueeze(1)

        for i in range(chunk_bsz):
            gen_ids = generated_tokens[i]
            text_gen = tokenizer.detokenize(gen_ids)
            full_text = chunk_prefixes[i] + text_gen

            reward_result = task.compute_total_reward(text_gen, chunk_refs[i])

            rew = reward_result["reward"]
            info = reward_result["reward_info"]

            valid_len = len(gen_ids)
            embs, lps = None, None

            if saved_lps[i] and valid_len > 0:
                lps = torch.stack(saved_lps[i][:valid_len])
            if loss_type == "ot" and saved_embs[i] and valid_len > 0:
                embs = torch.stack(saved_embs[i][:valid_len])

            all_episodes.append(
                Episode(chunk_prefixes[i], full_text, chunk_prefix_ids[i], gen_ids, rew, info, embs, lps)
            )

    return all_episodes


def update_policy(model, optimizer, episodes, config, pad_id, device, loss_type, beta):
    groups = defaultdict(list)
    for e in episodes:
        groups[e.prefix].append(e)
    for g in groups.values():
        rs = [e.reward for e in g]
        m, s = np.mean(rs), np.std(rs) + 1e-6
        for e in g:
            e.norm_reward = (e.reward - m) / s

    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    micro_bsz = config["training"]["micro_batch_size"]
    grad_steps = math.ceil(len(episodes) / micro_bsz)
    stats = defaultdict(float)

    for i in range(0, len(episodes), micro_bsz):
        batch = episodes[i : i + micro_bsz]
        lens = [len(e.prefix_token_ids) + len(e.generated_token_ids) for e in batch]
        max_len = max(lens)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        masks = torch.zeros((len(batch), max_len), device=device)
        model_dtype = model.dtype
        old_embs = torch.zeros((len(batch), max_len, model.config.hidden_size), device=device, dtype=model_dtype)
        old_lps = torch.zeros((len(batch), max_len), device=device, dtype=model_dtype)
        advs = []

        for k, e in enumerate(batch):
            pl, gl = len(e.prefix_token_ids), len(e.generated_token_ids)
            input_ids[k, : pl + gl] = torch.tensor(e.prefix_token_ids + e.generated_token_ids, device=device)
            masks[k, pl - 1 : pl + gl - 1] = 1.0
            advs.append(e.norm_reward)

            if loss_type == "ot" and e.embeddings is not None:
                val = min(len(e.embeddings), gl)
                if val > 0:
                    old_embs[k, pl - 1 : pl - 1 + val] = e.embeddings[:val].to(device).to(model_dtype)
            if e.old_log_probs is not None:
                val = min(len(e.old_log_probs), gl)
                if val > 0:
                    old_lps[k, pl - 1 : pl - 1 + val] = e.old_log_probs[:val].to(device).to(model_dtype)

        advs = torch.tensor(advs, device=device, dtype=model_dtype)
        outputs = model(input_ids, output_hidden_states=True)
        logits, h_new = outputs.logits, outputs.hidden_states[-1]

        logits_pred = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        mask_pred = masks[:, :-1]
        log_probs = torch.log_softmax(logits_pred, dim=-1)
        token_lps = torch.gather(log_probs, 2, targets.unsqueeze(2)).squeeze(2)

        pg_loss = -(token_lps * advs.unsqueeze(1) * mask_pred).sum() / (mask_pred.sum() + 1e-6)

        tr_loss = torch.tensor(0.0, device=device, dtype=model_dtype)

        if mask_pred.sum() > 0:
            if loss_type == "ot":
                ot_dist = robust_sinkhorn(
                    old_embs[:, :-1], h_new[:, :-1], mask_pred, window_size=config["training"]["ot_window"]
                )
                stats["ot_dist"] += ot_dist.mean().item()
                tr_loss = torch.clamp(ot_dist - config["training"].get("ot_target", 0.0), min=0.0).mean()

            kl_vals = old_lps[:, :-1] - token_lps
            stats["kl_dist"] += (kl_vals * mask_pred).sum().item() / (mask_pred.sum().item() + 1e-6)

            if loss_type == "kl":
                k_loss = torch.clamp(kl_vals - config["training"].get("kl_target", 0.0), min=0.0)
                tr_loss = (k_loss * mask_pred).sum() / (mask_pred.sum() + 1e-6)

            # Approx KL with Stability Clamp
            log_ratio = old_lps[:, :-1] - token_lps
            ratio = torch.exp(torch.clamp(log_ratio, max=10.0))
            approx_kl_vals = ratio - log_ratio - 1.0
            approx_kl_dist = (approx_kl_vals * mask_pred).sum() / (mask_pred.sum() + 1e-6)
            stats["approx_kl"] += approx_kl_dist.item()

            if loss_type == "approx_kl":
                tr_loss = approx_kl_dist

        loss = pg_loss + beta * tr_loss
        loss.backward()

        with torch.no_grad():
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
            mean_entropy = (entropy * mask_pred).sum() / (mask_pred.sum() + 1e-6)
            stats["entropy"] += mean_entropy.item()

        stats["loss"] += loss.item()

    nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
    optimizer.step()
    optimizer.zero_grad()
    return {k: v / grad_steps for k, v in stats.items()}


@torch.no_grad()
def evaluate(model, loader, tokenizer, task: BaseTask, config, device, loss_type):
    model.eval()
    accs, fmts = [], []

    for batch in loader:
        bsz = len(batch.prefix)

        input_ids = [torch.tensor(ids, device=device) for ids in batch.prefix_token_ids]

        max_len = max(len(t) for t in input_ids)
        padded_input = torch.full((bsz, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)

        for i, ids in enumerate(input_ids):
            seq_len = len(ids)
            padded_input[i, -seq_len:] = ids
            attention_mask[i, -seq_len:] = 1

        output_ids = model.generate(
            inputs=padded_input,
            attention_mask=attention_mask,
            max_new_tokens=config["training"]["max_gen_len"],
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for i in range(bsz):
            input_len = len(input_ids[i])
            gen_only = output_ids[i][input_len:]

            text_gen = tokenizer.detokenize(gen_only.tolist())

            reward_result = task.compute_total_reward(text_gen, batch.references[i])

            accs.append(reward_result["reward_info"]["answer_reward"])
            fmts.append(reward_result["reward_info"]["format_reward"])

    model.train()

    mean_acc = np.mean(accs) if accs else 0.0
    mean_fmt = np.mean(fmts) if fmts else 0.0

    return mean_acc, mean_fmt


def run_training(config: dict[str, Any], dataset_name: str, work_dir: Path):
    loss_type = config["training"].get("loss_type", "ot")
    beta = config["training"].get("beta", 0.1)
    logger.info(f"=== Dataset: {dataset_name} | Training Mode: {loss_type.upper()} ===")

    device = torch.device(config["model"]["device"])
    torch.manual_seed(config["training"]["random_seed"])

    original_model_path = config["model"]["pretrained_model_path"]
    logger.info(f"Requested Model: {original_model_path}")

    local_model_path = get_or_download_model(original_model_path)

    logger.info(f"Loading Tokenizer from: {local_model_path}")
    tokenizer = Tokenizer(local_model_path)

    logger.info(f"Loading Model Weights from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        dtype=torch.bfloat16 if config["model"]["dtype"] == "bfloat16" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    optimizer = MemoryEfficientAdamW(model.parameters(), lr=config["training"]["learning_rate"])

    logger.info(f"Initializing Task: {dataset_name}")
    task = get_task(dataset_name, tokenizer)

    train_ds = RLDataset(task, "train")
    test_ds = RLDataset(task, "test")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["num_questions_per_batch"],
        collate_fn=RLDataset.collate_fn,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["num_questions_per_batch"],
        collate_fn=RLDataset.collate_fn,
        shuffle=False,
    )

    tb_dir = work_dir / f"tb_{loss_type}_{datetime.now().strftime('%m%d_%H%M')}"
    writer = SummaryWriter(tb_dir)
    ckpt_dir = work_dir / "checkpoints"

    start_step = 0
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.glob("step_*.pt"))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))
            start_step = int(re.search(r"step_(\d+)", latest_ckpt.name).group(1))
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            logger.info(f"Resumed from step {start_step}")

    step = start_step
    n_samples = config["training"]["batch_size"] // config["training"]["num_questions_per_batch"]
    inference_bsz = config["training"].get("inference_batch_size", 32)

    logger.info("Starting Training...")

    eval_acc, fmt_acc = evaluate(model, test_loader, tokenizer, task, config, device, loss_type)
    logger.info(f" >> [Initial EVAL] Accuracy: {eval_acc:.2f}, format: {fmt_acc: .2f}")
    writer.add_scalar("Eval/Success", eval_acc, step)

    while True:
        for batch in train_loader:
            step += 1
            episodes = rollout(
                model,
                batch,
                tokenizer,
                task,
                config["training"]["max_gen_len"],
                n_samples,
                device,
                loss_type,
                inference_bsz,
            )
            raw_accs = [e.reward_info["answer_reward"] for e in episodes]
            tr_acc = np.mean(raw_accs)

            if step % 5 == 1:
                ep = episodes[0]
                debug_msg = (
                    f"\n[Step {step} Diagnostic]\n"
                    f"Prefix: {ep.prefix[:50]}...\n"
                    f"Generated: ...{ep.text[-100:].replace(chr(10), ' ')}\n"
                    f"Reward: {ep.reward:.4f} (Ans: {ep.reward_info['answer_reward']})"
                )
                logger.info(debug_msg)

            stats = update_policy(model, optimizer, episodes, config, tokenizer.pad_token_id, device, loss_type, beta)

            main_dist = stats.get("ot_dist", 0.0) if loss_type == "ot" else stats.get("kl_dist", 0.0)

            log_msg = (
                f"Step {step} | Loss: {stats['loss']:.4f} | "
                f"Acc: {tr_acc:.2f} | Dist: {main_dist:.4f} | Ent: {stats['entropy']:.4f}"
            )
            logger.info(log_msg)

            writer.add_scalar("Train/Loss", stats["loss"], step)
            writer.add_scalar("Train/Success", tr_acc, step)
            writer.add_scalar(f"Dist/{loss_type}", main_dist, step)

            if step % config["training"]["eval_interval"] == 0:
                eval_acc, fmt_acc = evaluate(model, test_loader, tokenizer, task, config, device, loss_type)
                logger.info(f" >> [Eval] Accuracy: {eval_acc:.2f}, format: {fmt_acc: .2f}")
                writer.add_scalar("Eval/Success", eval_acc, step)
                writer.add_scalar("Eval/Format", fmt_acc, step)

            if step % config["training"]["ckpt_save_interval"] == 0:
                p = ckpt_dir / f"step_{step}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p)
