"""Training-free looped inference for a frozen HF causal LM (arXiv:2605.23872).

Wraps a plain pretrained model (SmolLM2-360M here, no architecture conversion,
no fine-tuning) so that a contiguous mid-stack block of decoder layers is
re-applied K extra times at inference, via a forward hook on the last layer
of the block. Each extra pass uses a damped (forward-Euler) update:

    h_{i+1} = h_i + eta * (f(h_i) - h_i)

instead of naively feeding f(h_i) straight back in (eta=1 recovers naive
looping). K=0 is a no-op passthrough (base model behavior, hook attached but
inert) so the same code path can serve as the "base" arm of a comparison.

KV cache correctness: the block's *one* official pass (as HF already runs it
in the normal per-layer loop) writes to the model's real cache exactly as it
would without this wrapper. Each of the K replay passes runs against a
tensor-cloned copy of that real cache's block-layer entries -- so replay
attention reads real past context, but never mutates it -- then the copy is
discarded. (A generic `copy.deepcopy` of the whole Cache object blows Python's
recursion limit here, walking through torch storage internals from inside an
already-deep model forward call stack -- so we clone only the handful of
per-layer key/value tensors the replay actually touches.) This keeps
future-token attention unaffected by the extra passes' bookkeeping while
still letting every replay pass attend correctly. It also means the cached
KV for the current token reflects the pre-loop (single-pass) representation,
not the post-loop one -- a deliberate, documented approximation for this
cheap sanity check, not a paper-faithful recurrent-KV scheme.
"""

import dataclasses
import sys

import torch
from transformers.cache_utils import DynamicCache

# Replay passes call the block's layers from inside an already-deep model
# forward call stack (embedded in a forward hook, itself inside generate()'s
# own call chain); each layer call is ~20-30 Python frames, so a large block
# deep in the stack can approach the default limit of 1000.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))


@dataclasses.dataclass
class LoopConfig:
    start_idx: int
    end_idx: int  # inclusive
    k: int
    eta: float = 0.5


def _clone_cache_block(cache, start_idx, end_idx):
    """Cheap, non-recursive copy of a DynamicCache's entries for [start_idx, end_idx].

    Entries outside the block are shared by reference (never touched by
    replay layers); entries inside are real tensor clones so replay's
    `.update()` calls can't mutate the real cache.
    """
    new_cache = DynamicCache()
    new_cache._seen_tokens = cache._seen_tokens
    new_cache.key_cache = list(cache.key_cache)
    new_cache.value_cache = list(cache.value_cache)
    for i in range(start_idx, min(end_idx + 1, len(cache.key_cache))):
        new_cache.key_cache[i] = cache.key_cache[i].clone()
    for i in range(start_idx, min(end_idx + 1, len(cache.value_cache))):
        new_cache.value_cache[i] = cache.value_cache[i].clone()
    return new_cache


class _LoopHookHandle:
    def __init__(self, *handles):
        self._handles = handles

    def remove(self):
        for h in self._handles:
            h.remove()


def attach_loop_hook(model, cfg: LoopConfig):
    layers = model.model.layers
    block = layers[cfg.start_idx : cfg.end_idx + 1]
    # `Cache.update()` always appends -- it has no "replace" mode. So a replay
    # pass must start from the cache as it stood *before* the block's one
    # official pass appended this call's tokens, not from the (already grown)
    # cache captured after that pass -- otherwise each replay's update() call
    # appends a second copy of this call's tokens, doubling the block's cache
    # entries and desyncing them from the attention_mask/position_ids shapes
    # captured for the single pass-0 length. A pre-hook on the first block
    # layer snapshots that "ante-block" state once; every replay iteration
    # clones from that same fixed snapshot (never from a prior replay's own
    # growth) so each one appends exactly the one call's worth of tokens, same
    # as pass 0 did.
    #
    # The pre-hook lives on layers[start_idx] == block[0], and the post-hook
    # on layers[end_idx] == block[-1] (equal when the block is one layer) --
    # both are re-entered when replay calls the block's own layers, so both
    # need the same reentrancy guard.
    state = {"active": False, "pre_cache": None}

    def pre_hook(module, args, kwargs):
        if cfg.k == 0 or state["active"]:
            return None
        past_key_value = kwargs.get("past_key_value")
        state["pre_cache"] = (
            _clone_cache_block(past_key_value, cfg.start_idx, cfg.end_idx)
            if past_key_value is not None
            else None
        )
        return None

    def post_hook(module, args, kwargs, output):
        if cfg.k == 0 or state["active"]:
            return output
        state["active"] = True
        try:
            h = output[0]
            for _ in range(cfg.k):
                cache_copy = (
                    _clone_cache_block(state["pre_cache"], cfg.start_idx, cfg.end_idx)
                    if state["pre_cache"] is not None
                    else None
                )
                replay_kwargs = dict(kwargs)
                replay_kwargs["past_key_value"] = cache_copy
                x = h
                for layer in block:
                    x = layer(x, **replay_kwargs)[0]
                h = h + cfg.eta * (x - h)
            return (h,) + output[1:]
        finally:
            state["active"] = False
            state["pre_cache"] = None

    h1 = layers[cfg.start_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layers[cfg.end_idx].register_forward_hook(post_hook, with_kwargs=True)
    return _LoopHookHandle(h1, h2)


class LoopedModel:
    """Context manager: attaches the loop hook on enter, removes it on exit."""

    def __init__(self, model, cfg: LoopConfig):
        self.model = model
        self.cfg = cfg
        self._handle = None

    def __enter__(self):
        self._handle = attach_loop_hook(self.model, self.cfg)
        return self.model

    def __exit__(self, *exc):
        self._handle.remove()
        self._handle = None


@torch.no_grad()
def _self_test():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "HuggingFaceTB/SmolLM2-360M"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()

    ids = tok("The capital of France is", return_tensors="pt")

    base_logits = model(**ids).logits
    cfg0 = LoopConfig(start_idx=10, end_idx=15, k=0, eta=0.5)
    with LoopedModel(model, cfg0):
        looped_k0_logits = model(**ids).logits
    assert torch.allclose(base_logits, looped_k0_logits), "k=0 must be a no-op"

    cfg = LoopConfig(start_idx=10, end_idx=15, k=2, eta=0.5)
    with LoopedModel(model, cfg):
        prefill_logits = model(**ids).logits
        gen = model.generate(**ids, max_new_tokens=8, do_sample=False, use_cache=True)
        batch_ids = tok(["The capital of France is", "Two plus two equals"], return_tensors="pt", padding=True)
        batch_logits = model(**batch_ids).logits
    assert not torch.allclose(base_logits, prefill_logits), "k=2 should change logits"
    assert batch_logits.shape[0] == 2
    print("prefill OK, generate OK, batch OK:", tok.decode(gen[0]))
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
