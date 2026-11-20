"""Frozen Qwen3.5-family native cached execution with a conditional middle bypass."""
import copy
import hashlib
import importlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn
import transformers

SKIPS = (3, 7, 9, 11)
SOURCES = (8, 9, 10, 11, 12, 16, 24, 32, 40, 48)
GRID_SKIPS = (1, 2, 3, 4, 5, 6, 7, 11, 15, 23)
FIELDS = ("key_cache", "value_cache", "conv_states", "recurrent_states")


def emit(tag, **values):
    print("PB_" + tag + " " + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def chat_ids(tokenizer, messages):
    # Transformers 5.3 changed the default to BatchEncoding. Every caller here
    # needs one unbatched list, so make that return contract explicit.
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                      enable_thinking=False,return_dict=False)
    assert isinstance(ids,list) and all(isinstance(x,int) for x in ids)
    return ids


def check_tokenizer(tokenizer):
    messages=[dict(role='user',content='What is seven plus twelve?')]
    ids=chat_ids(tokenizer,messages)
    reference=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                            enable_thinking=False,return_dict=True)
    assert ids==reference['input_ids'] and len(ids)>1
    for choice in 'ABCD':
        encoded=tokenizer.encode(' '+choice,add_special_tokens=False)
        assert isinstance(encoded,list) and len(encoded)==1,(choice,encoded)
    emit('TOKENIZER_CONTRACT',chat_return='list[int]',explicit_dict_parity=True,mmlu_single_token_choices=True)


def load_model(path, rank, weight_inventory):
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    device = f"cuda:{rank}"
    config = json.loads((path/"config.json").read_text())
    assert config["model_type"] == "qwen3_5"
    weight_inventory=Path(weight_inventory)
    fingerprint=sha(weight_inventory)
    inventory=json.loads(weight_inventory.read_text())
    indexed=set(json.loads((path/'model.safetensors.index.json').read_text())['weight_map'].values())
    assert indexed=={record['file'] for record in inventory['files']},'weight inventory/index mismatch'
    for key,name in [('config_sha256','config.json'),('index_sha256','model.safetensors.index.json')]:
        if key in inventory:assert sha(path/name)==inventory[key],name
    # Full shard hashing is performed by fingerprint_model.py before launch.
    # Every worker verifies the recorded storage stamp without rereading 55 GB.
    for record in inventory['files']:
        status=(path/record['file']).stat()
        assert status.st_size==record['bytes'] and status.st_mtime_ns==record['mtime_ns'],record['file']
    tick = time.monotonic()
    model = getattr(transformers, config["architectures"][0]).from_pretrained(
        path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": device}, attn_implementation="sdpa").eval().requires_grad_(False)
    candidates = [model.model, getattr(model.model, "language_model", None), getattr(model, "language_model", None)]
    decoder = next(x for x in candidates if x is not None and hasattr(x, "layers") and hasattr(x, "norm"))
    assert len(decoder.layers) == 64 and decoder.config.hidden_size == 5120
    head = model.get_output_embeddings()
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    module = importlib.import_module(type(decoder).__module__)
    runtime = Path(module.__file__)
    assert hasattr(module, "Qwen3_5DynamicCache"), "unsupported native cache layout"
    evidence = dict(model=str(path), config_sha256=sha(path/"config.json"),
                    weight_inventory_sha256=fingerprint,
                    index_sha256=sha(path/"model.safetensors.index.json"),
                    runtime_sha256=sha(runtime), torch=str(torch.__version__),
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(rank),
                    gpu_uuid=str(torch.cuda.get_device_properties(rank).uuid),
                    fast_gdn=bool(getattr(module,"is_fast_path_available",False)),
                    seconds=time.monotonic()-tick)
    emit("MODEL", rank=rank, **evidence)
    return model, decoder, head, tokenizer, evidence


def fork_cache(cache):
    """FA append is immutable; the native GDN kernels mutate recurrent/conv state."""
    result = copy.copy(cache)
    for name in FIELDS:
        setattr(result, name, [x.clone() if x is not None and name in ("conv_states", "recurrent_states") else x
                               for x in getattr(cache, name)])
    return result


class SharedGate(nn.Module):
    def __init__(self, source=12, skips=SKIPS, width=1024):
        super().__init__()
        self.source, self.skips, self.width = source, tuple(skips), width
        self.net = nn.Sequential(nn.Linear(5120+len(skips), width), nn.ReLU(), nn.Dropout(.1), nn.Linear(width, 1))
        self.register_buffer("categories", torch.eye(len(skips)), persistent=False)

    def logits(self, hidden):
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1,keepdim=True).clamp_min(1e-12))
        shape = hidden.shape[:-1]
        h = hidden.unsqueeze(-2).expand(*shape,len(self.skips),5120)
        cat = self.categories.expand(*shape,len(self.skips),len(self.skips))
        return self.net(torch.cat((h,cat),-1)).squeeze(-1)


class Route:
    """Preserve each native block and cache; bypass only the selected middle blocks."""
    def __init__(self, decoder):
        self.decoder = decoder
        self.stack = ExitStack()
        self.active = False
        self.bounds = (0, 0)
        self.gate = None
        self.threshold = 1.
        self.force = None
        self.hidden = None
        self.selected = 0
        self.trace = None
        self.source = 12
        self.captures = None

    def __enter__(self):
        for index, layer in enumerate(self.decoder.layers):
            original = layer.forward
            if "forward" in layer.__dict__:
                self.stack.callback(setattr, layer, "forward", layer.__dict__["forward"])
            else:
                self.stack.callback(delattr, layer, "forward")
            def wrapped(hidden_states, *args, _i=index, _f=original, **kwargs):
                if self.bounds[0] <= _i < self.bounds[1]:
                    return hidden_states
                if self.trace is not None:
                    self.trace.append(_i+1)
                result = _f(hidden_states,*args,**kwargs)
                if self.captures is not None and _i+1 in SOURCES:
                    self.captures[_i+1] = result
                if _i == self.source-1:
                    self.hidden = result
                    if self.active:
                        if self.force is not None:
                            selected = self.force
                        else:
                            scores = self.gate.logits(result[:, -1]).sigmoid()[0]
                            widths = torch.tensor(self.gate.skips, device=scores.device)
                            selected = int(torch.where(scores >= self.threshold, widths, 0).max())
                        self.selected = selected
                        self.bounds = (self.source, self.source+selected)
                return result
            layer.forward = wrapped
        return self

    def __exit__(self,*error):
        return self.stack.__exit__(*error)

    def run(self, inputs, cache, position, *, source=12, skip=0, embedded=False, gate=None, threshold=1., force=None, trace=False, capture=False, use_cache=True):
        self.source = gate.source if gate is not None else source
        self.bounds = (0,self.source+skip) if embedded else (self.source,self.source+skip)
        self.active = gate is not None or force is not None
        self.gate, self.threshold, self.force = gate, threshold, force
        self.hidden = None
        self.selected = skip
        self.trace = [] if trace else None
        self.captures = {} if capture else None
        try:
            result = self.decoder(**{"inputs_embeds" if embedded else "input_ids":inputs},
                past_key_values=cache, cache_position=position, use_cache=use_cache, output_hidden_states=False)
            return result.last_hidden_state, result.past_key_values, self.hidden, self.selected, self.trace
        finally:
            self.active = False
            self.bounds = (0,0)


def assert_cache_equal(a,b,depths):
    for i in depths:
        for field in FIELDS:
            x,y = getattr(a,field)[i], getattr(b,field)[i]
            assert (x is None)==(y is None)
            assert x is None or torch.equal(x,y), (field,i)


@torch.inference_mode()
def parity(model, decoder, head, tokenizer, device, source=12, skips=SKIPS):
    ids = tokenizer.encode("A short arithmetic problem: if there are twelve boxes and each has seven apples, how many apples are there?",add_special_tokens=False)
    ids = torch.tensor([ids],device=device)
    first = min(16,ids.shape[1]-8)
    checks=[]
    with Route(decoder) as route:
        _,cache,_,_,_=route.run(ids[:,:first],None,torch.arange(first,device=device))
        for p in range(first,first+3):
            before=fork_cache(cache)
            final,cache,hidden,_,trace=route.run(ids[:,p:p+1],cache,torch.tensor([p],device=device),source=source,trace=True)
            assert trace==list(range(1,65))
            reference=head(final[:,-1]).float()
            native=decoder(input_ids=ids[:,:p+1],use_cache=False).last_hidden_state[:,-1]
            native_logits=head(native).float()
            assert int(reference.argmax(-1))==int(native_logits.argmax(-1))
            a,b=reference.log_softmax(-1),native_logits.log_softmax(-1)
            kl=float((a.exp()*(a-b)).sum())
            assert kl < .01, kl
            for skip in skips:
                direct_cache,shared_cache=fork_cache(before),fork_cache(before)
                direct,direct_cache,_,_,direct_trace=route.run(ids[:,p:p+1],direct_cache,torch.tensor([p],device=device),source=source,skip=skip,trace=True)
                shared,shared_cache,_,_,shared_trace=route.run(hidden,shared_cache,torch.tensor([p],device=device),source=source,skip=skip,embedded=True,trace=True)
                assert torch.equal(direct,shared), (p,skip)
                assert direct_trace==[d for d in range(1,65) if not source < d <= source+skip]
                assert shared_trace==list(range(source+1+skip,65))
                assert_cache_equal(before,shared_cache,range(source,source+skip))
                assert_cache_equal(direct_cache,shared_cache,range(source+skip,64))
            checks.append(dict(position=p,full_top1=True,full_kl=kl,all_direct_shared_bitwise=True,skipped_states_frozen=True))
        # Exercise re-entry after missing FA tokens and frozen GDN steps.
        _,mixed,_,_,_=route.run(ids[:,:first],None,torch.arange(first,device=device))
        schedule=(max(skips),0,min(skips),skips[len(skips)//2],max(skips),0)
        for offset,skip in enumerate(schedule):
            p=first+offset
            before=fork_cache(mixed)
            _,mixed,_,selected,trace=route.run(ids[:,p:p+1],mixed,torch.tensor([p],device=device),source=source,force=skip,trace=True)
            assert selected==skip
            assert trace==[d for d in range(1,65) if not source < d <= source+skip]
            assert_cache_equal(before,mixed,range(source,source+skip))
    official=model(input_ids=ids[:,:first],use_cache=False,logits_to_keep=1).logits[:,-1].argmax(-1)
    native=head(decoder(input_ids=ids[:,:first],use_cache=False).last_hidden_state[:,-1]).argmax(-1)
    assert torch.equal(official,native)
    return dict(native_official_top1=True,checks=checks,mixed_reentry_and_sparse_cache=True)
