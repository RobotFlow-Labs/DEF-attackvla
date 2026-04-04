import functools
import gc
import inspect
import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Optional, Union, TypeVar, Generic
import transformers

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

# Common configuration used by both GCG implementations
@dataclass
class GCGConfig:
    num_steps: int = 500
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = 64
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = True
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = 42
    verbosity: str = "INFO"
    unnorm_key: str = None
    use_trace: bool = False
    use_pi0fast: bool = False
    as_suffix: bool = False

# Unified result class for single and multi-model GCG
@dataclass
class GCGResult:
    best_loss: float
    # For single model: string, for multi-model: list of strings (one per model)
    best_strings: Union[str, List[str]]
    losses: List[float]
    # For single model: list of strings, for multi-model: list of lists of strings
    strings: Union[List[str], List[List[str]]]
    # For single model: string, for multi-model: list of strings
    perfect_match_strs: Union[str, List[str]] = ''
    # For single model: single tensor, for multi-model: list of tensors
    perfect_match_ids: Optional[Tensor] = None
    # For single model: single tensor, for multi-model: list of tensors
    num_steps_taken: int = 0
    total_time: float = 0.0
    init_str_length: int = 0

    @property
    def perfect_match_optim_str(self) -> str:
        """For backward compatibility with single-model case."""
        if isinstance(self.perfect_match_optim_strs, list):
            return self.perfect_match_optim_strs[0] if self.perfect_match_optim_strs else ''
        return self.perfect_match_optim_strs

    @property
    def best_string(self) -> str:
        """For backward compatibility with single-model case."""
        if isinstance(self.best_strings, list):
            return self.best_strings[0] if self.best_strings else ''
        return self.best_strings

class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer=None, processors=None):
        message = "buffer:"
        for loss, ids in self.buffer:
            if processors:
                # Multi-model case
                decoded_list = [proc.tokenizer.batch_decode(ids)[0] for proc in processors]
                message += f"\nloss: {loss} | strings: {decoded_list}"
            else:
                # Single-model case
                optim_str = tokenizer.batch_decode(ids)[0]
                optim_str = optim_str.replace("\\", "\\\\")
                optim_str = optim_str.replace("\n", "\\n")
                message += f"\nloss: {loss}" + f" | string: {optim_str}"
        return message

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = None,
):
    """
    Returns `search_width` new token sequences by sampling from the gradient signal.

    Args:
        ids (Tensor): shape (n_optim_tokens,)
        grad (Tensor): shape (n_optim_tokens, vocab_size)
        search_width (int): how many candidate sequences to create
        topk (int): sample from the topk positions in each token's gradient
        n_replace (int): how many token positions to update per sequence
        not_allowed_ids (Tensor): optional mask of disallowed IDs
    
    Returns:
        (Tensor): shape (search_width, n_optim_tokens)
    """
    # Move not_allowed_ids to the same device as grad if it exists
    if not_allowed_ids is not None:
        not_allowed_ids = not_allowed_ids.to(grad.device)
        grad[:, not_allowed_ids] = float("inf")
    
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    topk_ids = (-grad).topk(topk, dim=1).indices

    # Randomly pick n_replace positions to mutate
    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    # For each chosen position, randomly pick among the topk new IDs
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    # Replace the chosen positions with the chosen new token IDs
    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids

def filter_ids_single(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """
    Filters out sequences that become different when re-tokenized. 
    This ensures token IDs are stable under decode-then-encode.
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []
    for i in range(len(ids_decoded)):
        reencoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False)["input_ids"].to(ids.device)[0]
        if torch.equal(ids[i], reencoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are identical after decoding and re-encoding. "
            "Either set `filter_ids=False` or use a different emphasis on `optim_str_init`."
        )
    
    return torch.stack(filtered_ids)

def filter_ids_multi(ids: Tensor, processors: List[transformers.AutoProcessor]):
    """
    Filters out candidate sequences that become different when re-tokenized
    by all provided tokenizers.
    """
    filtered_ids = []
    for i in range(len(ids)):
        valid = True
        for proc in processors:
            decoded = proc.tokenizer.batch_decode(ids[i].unsqueeze(0))[0]
            reencoded = proc.tokenizer(decoded, return_tensors="pt", add_special_tokens=False)["input_ids"].to(ids.device)[0]
            if not torch.equal(ids[i], reencoded):
                valid = False
                break
        if valid:
            filtered_ids.append(ids[i])
    
    if not filtered_ids:
        raise RuntimeError(
            "No token sequences are identical after decoding and re-encoding for all models. "
            "Either set `filter_ids=False` or use a different emphasis on `optim_str_init`."
        )
    
    return torch.stack(filtered_ids)

def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(nonascii_toks, device=device)

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
   return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user errorw
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator
