import dataclasses
import math
import torch
import re
import pandas as pd
import numpy as np
import torch.nn.functional as F
import yaml
import argparse
import logging
import sys
import os
from datetime import datetime
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import snapshot_download

# --- Setup Arguments ---
parser = argparse.ArgumentParser(description="Training Script")
parser.add_argument("--config", type=str, default="config_math.yaml", help="Path to the config file")
parser.add_argument("--dataset_name", type=str, default="gsm8k", choices=["gsm8k", "math", "countdown"], help="Dataset to use")
args = parser.parse_args()

# --- Setup Workspace & Logging ---
config_path = Path(args.config)
config_name = config_path.stem
work_dir = Path("runs") / f"{config_name}_{args.dataset_name}"
work_dir.mkdir(parents=True, exist_ok=True)

log_file_path = work_dir / "training.log"
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# --- Templates ---
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

MATH_TEMPLATE = (
    "Solve the following problem. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> 42 </answer>.\n\n"
    "--- Example ---\n"
    "Problem: Joy has 5 balls. He buys 2 more. How many balls does he have?\n"
    "<think>\n"
    "Joy starts with 5 balls.\n"
    "He buys 2 more balls.\n"
    "5 + 2 = 7.\n"
    "</think>\n"
    "<answer> 7 </answer>\n\n"
    "--- Your Task ---\n"
    "Problem: {question}"
)

RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

# --- Helper: Model Download ---
def get_or_download_model(model_path_or_id: str) -> str:
    """
    Check if model exists locally. If not, download to ./models/{model_name}.
    Returns the local path to the model.
    """
    # 1. If it's already a valid local path, return class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 强制设置为左填充，这对 Decoder-only 模型生成至关重要
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_tokenit
    if os.path.exists(model_path_or_id):
        logger.info(f"Model found locally at: {model_path_or_id}")
        return model_path_or_id

    # 2. Setup local models directory
    current_dir = Path.cwd()
    models_root = current_dir / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    # 3. Construct target directory name (e.g., Qwen/Qwen2.5 -> Qwen2.5)
    model_name = model_path_or_id.split("/")[-1]
    local_model_path = models_root / model_name

    # 4. Check if already downloaded (simple check: if dir exists and has config.json)
    if local_model_path.exists() and (local_model_path / "config.json").exists():
        logger.info(f"Model already downloaded at: {local_model_path}")
        return str(local_model_path)
    
    logger.info(f"Model not found locally. Downloading {model_path_or_id} to {local_model_path}...")
    try:
        snapshot_download(
            repo_id=model_path_or_id,
            local_dir=local_model_path,
            local_dir_use_symlinks=False, # Make it standalone
            resume_download=True
        )
        logger.info(f"Download complete.")
        return str(local_model_path)
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise e

def extract_answer_content(response: str) -> Optional[str]:
    answer_regex = r"<answer>(.*?)<\/answer>"
    match = re.search(answer_regex, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if "answer is" in response.lower():
        return response.lower().split("answer is")[-1]
    
    return None

def extract_last_number(text: str) -> Optional[float]:
    text = text.replace(',', '')
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None

def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]
    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"
    
    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)
    
    if full_format_match: return 1.0
    reward = 0.0
    if think_match: reward += 0.1
    if answer_match: reward += 0.5
    return reward

def math_reward_function(response: str, ground_truth: str) -> float:
    model_answer_content = extract_answer_content(response)
    if not model_answer_content:
        return 0.0

    model_val = extract_last_number(model_answer_content)
    
    if "####" in ground_truth:
        gt_clean = ground_truth.split("####")[-1].strip()
    else:
        gt_clean = ground_truth
    
    gt_val = extract_last_number(gt_clean)

    if model_val is not None and gt_val is not None:
        if abs(model_val - gt_val) < 1e-4:
            return 1.0
    
    return 0.0

def reward_function(response: str, ground_truth: str, end_token: str = None) -> Dict[str, Any]:
    fmt_rew = format_reward_function("<think>" + response, end_token)
    ans_rew = math_reward_function(response, ground_truth)
    noise = len(response) * 1e-6
    total_reward = fmt_rew * 0.1 + ans_rew * 1.0 + noise
    
    return {
        "reward": total_reward,
        "reward_info": {
            "format_reward": fmt_rew,
            "answer_reward": ans_rew,
        },
    }

class MemoryEfficientAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

class Tokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left" 
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

    def encode_chat_with_response_prompt(self, messages: List[Dict[str, str]], prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + prompt

    def tokenize(self, text: str):
        return self.tokenizer(text, add_special_tokens=False)

    def detokenize(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

@dataclass
class MiniBatch:
    prefix: List[str]
    prefix_token_ids: List[List[int]]
    references: List[str]

@dataclass
class Episode:
    prefix: str
    text: str
    prefix_token_ids: List[int]
    generated_token_ids: List[int]
    reward: float
    reward_info: Dict[str, float]
    embeddings: Optional[torch.Tensor] = None 
    old_log_probs: Optional[torch.Tensor] = None 

class ReasoningDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, dataset_name: str, split: str = "train"):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.data = []
        
        if self.dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split=split)
            for item in ds:
                self.data.append({
                    "question": item["question"],
                    "ground_truth": item["answer"]
                })
        elif self.dataset_name == "math":
            ds = load_dataset("hendrycks/competition_math", split=split)
            for item in ds:
                self.data.append({
                    "question": item["problem"],
                    "ground_truth": item["solution"]
                })
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Loaded {len(self.data)} examples from {dataset_name} ({split})")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        user_msg = MATH_TEMPLATE.format(question=item["question"])
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_msg}],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix, 
            "ids": tokens['input_ids'], 
            "reference": item["ground_truth"]
        }

    @staticmethod
    def collate_fn(batch):
        return MiniBatch(
            prefix=[b["prefix"] for b in batch],
            prefix_token_ids=[b["ids"] for b in batch],
            references=[b["reference"] for b in batch]
        )

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
        v = -torch.logsumexp(log_K.transpose(1,2) + u.unsqueeze(1), 2).masked_fill(~row_mask, 0.0)
    P = torch.exp(u.unsqueeze(2) + v.unsqueeze(1) + log_K) * valid.float()
    return (P * cost).sum((1,2)) / (mask.sum(1) + 1e-6)

@torch.no_grad()
def rollout(model, batch, tokenizer, max_gen_len, num_samples, device, loss_type="ot"):

    total_bsz = len(batch.prefix) * num_samples
    all_prefix_ids = [ids for ids in batch.prefix_token_ids for _ in range(num_samples)]
    all_references = [r for r in batch.references for _ in range(num_samples)]
    all_prefixes = [p for p in batch.prefix for _ in range(num_samples)]
    
    inference_batch_size = 64 # 64 or 32 
    
    all_episodes = []
    
    for start_idx in range(0, total_bsz, inference_batch_size):
        end_idx = min(start_idx + inference_batch_size, total_bsz)
        chunk_bsz = end_idx - start_idx
        
        chunk_prefix_ids = all_prefix_ids[start_idx:end_idx]
        chunk_prefixes = all_prefixes[start_idx:end_idx]
        chunk_refs = all_references[start_idx:end_idx]
        
        max_p_len = max(len(p) for p in chunk_prefix_ids)
        input_ids = torch.full((chunk_bsz, max_p_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        
        for i, p in enumerate(chunk_prefix_ids): 
            l = len(p)
            input_ids[i, -l:] = torch.tensor(p, device=device)
            
        saved_embs = [[] for _ in range(chunk_bsz)]
        saved_lps = [[] for _ in range(chunk_bsz)]
        generated_tokens = [[] for _ in range(chunk_bsz)]
        finished = torch.zeros(chunk_bsz, dtype=torch.bool, device=device)
        past_key_values = None
        curr_input = input_ids
        
        for step in range(max_gen_len):
            outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :] 
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            
            lp = torch.log_softmax(next_logits, dim=-1)
            sel_lp = torch.gather(lp, 1, next_token.unsqueeze(1)).squeeze(1).float().cpu()
            for i in range(chunk_bsz):
                if not finished[i]: saved_lps[i].append(sel_lp[i])

            if loss_type == "ot":
                h = outputs.hidden_states[-1][:, -1, :].float().cpu()
                for i in range(chunk_bsz):
                    if not finished[i]: saved_embs[i].append(h[i])
            
            finished |= (next_token == tokenizer.eos_token_id)
            for i in range(chunk_bsz):
                if not finished[i]: generated_tokens[i].append(next_token[i].item())
            
            if finished.all(): break
            curr_input = next_token.unsqueeze(1)
            
        for i in range(chunk_bsz):
            gen_ids = generated_tokens[i]
            text_gen = tokenizer.detokenize(gen_ids)
            full_text = chunk_prefixes[i] + text_gen
            
            reward_result = reward_function(text_gen, chunk_refs[i], tokenizer.eos_token)
            rew = reward_result["reward"]
            info = reward_result["reward_info"]
            
            valid_len = len(gen_ids)
            embs, lps = None, None
            
            if saved_lps[i] and valid_len > 0:
                lps = torch.stack(saved_lps[i][:valid_len])
            if loss_type == "ot" and saved_embs[i] and valid_len > 0:
                embs = torch.stack(saved_embs[i][:valid_len])
                
            all_episodes.append(Episode(chunk_prefixes[i], full_text, chunk_prefix_ids[i], gen_ids, rew, info, embs, lps))
            
    return all_episodes

def update_policy(model, optimizer, episodes, config, pad_id, device, loss_type, beta):
    groups = defaultdict(list)
    for e in episodes: groups[e.prefix].append(e)
    for g in groups.values():
        rs = [e.reward for e in g]
        m, s = np.mean(rs), np.std(rs) + 1e-6
        for e in g: e.norm_reward = (e.reward - m) / s
        
    episodes.sort(key=lambda x: len(x.prefix_token_ids)+len(x.generated_token_ids))
    micro_bsz = config["training"]["micro_batch_size"]
    grad_steps = math.ceil(len(episodes)/micro_bsz)
    stats = defaultdict(float)
    
    for i in range(0, len(episodes), micro_bsz):
        batch = episodes[i:i+micro_bsz]
        lens = [len(e.prefix_token_ids)+len(e.generated_token_ids) for e in batch]
        max_len = max(lens)
        
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        masks = torch.zeros((len(batch), max_len), device=device)
        model_dtype = model.dtype
        old_embs = torch.zeros((len(batch), max_len, model.config.hidden_size), device=device, dtype=model_dtype)
        old_lps = torch.zeros((len(batch), max_len), device=device, dtype=model_dtype)
        advs = []
        
        for k, e in enumerate(batch):
            pl, gl = len(e.prefix_token_ids), len(e.generated_token_ids)
            input_ids[k, :pl+gl] = torch.tensor(e.prefix_token_ids + e.generated_token_ids, device=device)
            masks[k, pl-1 : pl+gl-1] = 1.0
            advs.append(e.norm_reward)
            
            if loss_type == "ot" and e.embeddings is not None:
                val = min(len(e.embeddings), gl)
                if val > 0: old_embs[k, pl-1 : pl-1+val] = e.embeddings[:val].to(device).to(model_dtype)
            if e.old_log_probs is not None:
                val = min(len(e.old_log_probs), gl)
                if val > 0: old_lps[k, pl-1 : pl-1+val] = e.old_log_probs[:val].to(device).to(model_dtype)
                
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
                ot_dist = robust_sinkhorn(old_embs[:, :-1], h_new[:, :-1], mask_pred, window_size=config["training"]["ot_window"])
                stats["ot_dist"] += ot_dist.mean().item()
                tr_loss = torch.clamp(ot_dist - config["training"].get("ot_target", 0.0), min=0.0).mean()
            
            kl_vals = old_lps[:, :-1] - token_lps
            stats["kl_dist"] += (kl_vals * mask_pred).sum().item() / (mask_pred.sum().item() + 1e-6)
            
            if loss_type == "kl":
                 k_loss = torch.clamp(kl_vals - config["training"].get("kl_target", 0.0), min=0.0)
                 tr_loss = (k_loss * mask_pred).sum() / (mask_pred.sum() + 1e-6)

            log_ratio = old_lps[:, :-1] - token_lps
            ratio = torch.exp(log_ratio)
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
    return {k: v/grad_steps for k,v in stats.items()}

@torch.no_grad()
def evaluate(model, loader, tokenizer, config, device, loss_type):
    model.eval()
    accs, fmts = [], []
    
    for batch in loader:
        bsz = len(batch.prefix)
        input_ids = [torch.tensor(ids, device=device) for ids in batch.prefix_token_ids]

        max_len = max(len(t) for t in input_ids)
        padded_input = torch.full((bsz, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        
        for i, ids in enumerate(input_ids):
            l = len(ids)
            padded_input[i, -l:] = ids
            attention_mask[i, -l:] = 1

        output_ids = model.generate(
            inputs=padded_input,
            attention_mask=attention_mask,
            max_new_tokens=config["training"]["max_gen_len"],
            do_sample=True,
            temperature=1.0, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        for i in range(bsz):
            input_len = len(input_ids[i])
            gen_only = output_ids[i][input_len:] 
            
            text_gen = tokenizer.detokenize(gen_only.tolist())
            
            reward_result = reward_function(text_gen, batch.references[i], tokenizer.eos_token)
            accs.append(reward_result["reward_info"]["answer_reward"])
            fmts.append(reward_result["reward_info"]["format_reward"])

    model.train()
    
    mean_acc = np.mean(accs) if accs else 0.0
    mean_fmt = np.mean(fmts) if fmts else 0.0
    
    return mean_acc, mean_fmt

def main():
    logger.info(f"Loading config from: {args.config}")
    with open(args.config) as f: config = yaml.safe_load(f)
    
    loss_type = config["training"].get("loss_type", "ot") 
    beta = config["training"].get("beta", 0.1)
    dataset_name = args.dataset_name
    logger.info(f"=== Dataset: {dataset_name} | Training Mode: {loss_type.upper()} ===")
    
    device = torch.device(config["model"]["device"])
    torch.manual_seed(config["training"]["random_seed"])
    
    original_model_path = config['model']['pretrained_model_path']
    logger.info(f"Requested Model: {original_model_path}")
    
    local_model_path = get_or_download_model(original_model_path)
    
    logger.info(f"Loading Tokenizer from: {local_model_path}")
    tokenizer = Tokenizer(local_model_path)
    
    logger.info(f"Loading Model Weights from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, 
        dtype=torch.bfloat16 if config["model"]["dtype"]=="bfloat16" else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    optimizer = MemoryEfficientAdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    train_ds = ReasoningDataset(tokenizer, dataset_name, "train")
    test_ds = ReasoningDataset(tokenizer, dataset_name, "test")
    
    train_loader = DataLoader(train_ds, batch_size=config["training"]["num_questions_per_batch"], 
                              collate_fn=ReasoningDataset.collate_fn, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["num_questions_per_batch"], 
                             collate_fn=ReasoningDataset.collate_fn, shuffle=False)
    
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
    
    logger.info("Starting Training...")
    
    while True:
        for batch in train_loader:
            step += 1
            episodes = rollout(model, batch, tokenizer, config["training"]["max_gen_len"], n_samples, device, loss_type)
            
            raw_accs = [e.reward_info["answer_reward"] for e in episodes]
            tr_acc = np.mean(raw_accs)
            
            if step % 5 == 1:
                ep = episodes[0]
                debug_msg = (f"\n[Step {step} Diagnostic]\n"
                             f"Prefix: {ep.prefix[:50]}...\n"
                             f"Generated: ...{ep.text[-100:].replace(chr(10), ' ')}\n"
                             f"Reward: {ep.reward:.4f} (Ans: {ep.reward_info['answer_reward']})")
                logger.info(debug_msg)

            stats = update_policy(model, optimizer, episodes, config, tokenizer.pad_token_id, device, loss_type, beta)
            
            main_dist = stats.get('ot_dist', 0.0) if loss_type == 'ot' else stats.get('approx_kl', stats.get('kl_dist', 0.0))
            
            log_msg = (
                f"Step {step} | Loss: {stats['loss']:.4f} | "
                f"Acc: {tr_acc:.2f} | Dist: {main_dist:.4f} | Ent: {stats['entropy']:.4f}"
            )
            logger.info(log_msg)
            
            writer.add_scalar("Train/Loss", stats["loss"], step)
            writer.add_scalar("Train/Success", tr_acc, step)
            writer.add_scalar(f"Dist/{loss_type}", main_dist, step)
            
            if step % config["training"]["eval_interval"] == 0:
                eval_acc, _ = evaluate(model, test_loader, tokenizer, config, device, loss_type)
                logger.info(f" >> [EVAL] Accuracy: {eval_acc:.2f}")
                writer.add_scalar("Eval/Success", eval_acc, step)
                
            if step % config["training"]["ckpt_save_interval"] == 0:
                p = ckpt_dir / f"step_{step}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), p)

if __name__ == "__main__":
    main()