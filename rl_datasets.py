import abc
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def extract_xml_answer(text: str) -> Optional[str]:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_latex_number(text: str) -> Optional[float]:
    if not text: return None
    text = text.replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        pass

    match = re.search(r"\\frac\{([0-9\.]+)\}\{([0-9\.]+)\}", text)
    if match:
        try:
            return float(match.group(1)) / float(match.group(2))
        except: pass
    return None


class BaseTask(abc.ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.system_prompt = (
            "You are a helpful assistant. You first think about the reasoning process "
            "in your mind and then provide the user with the answer."
        )

    @abc.abstractmethod
    def load_data(self, split: str) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def make_prompt(self, item: Dict[str, Any]) -> str:
        pass
    
    @abc.abstractmethod
    def get_reference(self, item: Dict[str, Any]) -> Any:
        pass

    @abc.abstractmethod
    def reward_function(self, response: str, reference: Any) -> float:
        pass

    def format_reward(self, response: str) -> float:
        think_match = re.search(r"<think>.*?<\/think>", response, re.DOTALL)
        answer_match = re.search(r"<answer>.*?<\/answer>", response, re.DOTALL)
        reward = 0.0
        if think_match: reward += 0.1
        if answer_match: reward += 0.5
        return reward

    def compute_total_reward(self, response: str, reference: Any) -> Dict[str, Any]:
        fmt_rew = self.format_reward("<think>" + response)
        ans_rew = self.reward_function(response, reference)
        
        noise = len(response) * 1e-6
        
        return {
            "reward": fmt_rew * 0.1 + ans_rew * 1.0 + noise,
            "reward_info": {
                "format_reward": fmt_rew,
                "answer_reward": ans_rew
            }
        }

    def _apply_chat_template(self, user_content: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True
        )
        return text + "Let me solve this step by step.\n<think>"


class GSM8KTask(BaseTask):
    def load_data(self, split: str) -> List[Dict[str, Any]]:
        ds = load_dataset("gsm8k", "main", split=split)
        return list(ds)

    def make_prompt(self, item: Dict[str, Any]) -> str:
        template = (
            "Solve the following problem. "
            "Show your work in <think> </think> tags. "
            "And return the final answer in <answer> </answer> tags, for example <answer> 42 </answer>.\n\n"
            "Problem: {question}"
        )
        return self._apply_chat_template(template.format(question=item["question"]))

    def get_reference(self, item: Dict[str, Any]) -> str:
        return item["answer"]

    def reward_function(self, response: str, reference: str) -> float:
        model_ans = extract_xml_answer(response)
        if not model_ans: return 0.0
        
        val_model = parse_latex_number(model_ans)
        
        gt_clean = reference.split("####")[-1].strip()
        val_gt = parse_latex_number(gt_clean)
        
        if val_model is not None and val_gt is not None:
            if abs(val_model - val_gt) < 1e-4:
                return 1.0
        return 0.0


class MATHTask(BaseTask):
    def load_data(self, split: str) -> List[Dict[str, Any]]:
        ds = load_dataset("lighteval/MATH", "all", split=split)
        acceptable_levels = ["Level 1", "Level 2", "Level 3"]
        
        data = []
        for item in ds:
            lvl = item.get("level")
            if lvl in acceptable_levels:
                data.append(item)
        return data

    def make_prompt(self, item: Dict[str, Any]) -> str:
        q = item.get("problem", item.get("question"))
        template = (
            "Solve the following math problem. "
            "Show your work in <think> </think> tags. "
            "Return the final answer in <answer> </answer> tags.\n\n"
            "Problem: {question}"
        )
        return self._apply_chat_template(template.format(question=q))

    def get_reference(self, item: Dict[str, Any]) -> str:
        return item.get("solution", item.get("answer"))

    def reward_function(self, response: str, reference: str) -> float:
        model_ans = extract_xml_answer(response)
        if not model_ans: return 0.0

        def extract_boxed(text):
            idx = text.find("\\boxed{")
            if idx == -1: return text
            idx += 7
            balance = 1
            for i in range(idx, len(text)):
                if text[i] == "{": balance += 1
                elif text[i] == "}": balance -= 1
                if balance == 0: return text[idx:i]
            return text

        gt_val_str = extract_boxed(reference)
        
        val_model = parse_latex_number(model_ans)
        val_gt = parse_latex_number(gt_val_str)

        if val_model is not None and val_gt is not None:
            if abs(val_model - val_gt) < 1e-4:
                return 1.0
        
        if str(model_ans).strip() == str(gt_val_str).strip():
            return 1.0
            
        return 0.0


class CountdownTask(BaseTask):
    def __init__(self, tokenizer, data_path: str = "Countdown-Tasks-3to4"):
        super().__init__(tokenizer)
        self.data_path = data_path

    def load_data(self, split: str) -> List[Dict[str, Any]]:
        path = Path(self.data_path) / "data"
        if not path.exists():
            logger.warning(f"Countdown data path {path} not found.")
            return []
            
        df = pd.read_parquet(path)
        test_size = 512
        if split == "train":
            data = df.iloc[:-test_size].to_dict('records')
        else:
            data = df.iloc[-test_size:].to_dict('records')
        return data

    def make_prompt(self, item: Dict[str, Any]) -> str:
        template = (
            "Using the numbers {numbers}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
            "Show your work in <think> </think> tags. "
            "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
        )
        return self._apply_chat_template(template.format(numbers=item["nums"], target=item["target"]))

    def get_reference(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {"nums": item["nums"], "target": item["target"]}

    def reward_function(self, response: str, reference: Dict[str, Any]) -> float:
        model_ans = extract_xml_answer(response)
        if not model_ans: return 0.0
        if "=" in model_ans: model_ans = model_ans.split("=")[0].strip()

        if not re.match(r"^[0-9+\-*/() ]+$", model_ans): return 0.0

        nums = reference["nums"]
        target = reference["target"]
        
        used_nums = [int(n) for n in re.findall(r"\d+", model_ans)]
        if sorted(used_nums) != sorted(nums): return 0.0
        
        try:
            result = eval(model_ans, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0
        except:
            pass
        return 0.0


class RLDataset(Dataset):
    def __init__(self, task: BaseTask, split: str):
        self.task = task
        self.data = task.load_data(split)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prefix = self.task.make_prompt(item)
        reference = self.task.get_reference(item)
        
        # Tokenize prefix
        tokens = self.task.tokenizer.tokenize(prefix)
        
        return {
            "prefix": prefix,
            "ids": tokens['input_ids'],
            "reference": reference
        }

    @staticmethod
    def collate_fn(batch):
        from dataclasses import dataclass
        @dataclass
        class MiniBatch:
            prefix: List[str]
            prefix_token_ids: List[List[int]]
            references: List[Any]

        return MiniBatch(
            prefix=[b["prefix"] for b in batch],
            prefix_token_ids=[b["ids"] for b in batch],
            references=[b["reference"] for b in batch]
        )

def get_task(task_name: str, tokenizer) -> BaseTask:
    task_map = {
        "gsm8k": GSM8KTask,
        "math": MATHTask,
        "countdown": CountdownTask
    }
    task_class = task_map.get(task_name.lower())
    if not task_class:
        raise ValueError(f"Unknown task: {task_name}")
    return task_class(tokenizer)

