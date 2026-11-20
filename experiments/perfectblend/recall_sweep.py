"""Recall at high precision for the source-32 early-exit gate, on already captured features.

Inputs are the frozen 27B capture r1 shards (layer-32 hidden, native one-step
counterfactual safety labels for widths 5/6/7/11) and the deployed five-class
classifier. Model selection uses a development slice of the training
conversations only; thresholds use the calibration split; heldout is reported.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 20261120
SKIPS = (5, 6, 7, 11)
ACTIONS = (0, *SKIPS)
WIDTHS = np.asarray(ACTIONS)
GOALS = (0.99, 0.995, 0.998)
SPLITS = ("train", "calibration", "heldout")
EXPECTED = dict(train=(10000, 79865), calibration=(1000, 7993), heldout=(1000, 7984))
DEPLOYED_SHA = "47b94a7b12155ad1a58a773fb2326cc2c28bc2129d25a72752a2a10374b46596"
MINIMUM_ACCEPTED = 32


def emit(tag, **values):
    print("PB_RECALL_" + tag + " " + json.dumps(values, allow_nan=False), flush=True)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------- data


def chat_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    assert isinstance(ids, list) and all(isinstance(x, int) for x in ids)
    return ids


def window(tokenizer, row, count=8, context=1024):
    """Identical to native/pipeline.py window(); recomputed to recover input tokens."""
    messages = []
    role_map = dict(human="user", gpt="assistant", system="system", user="user", assistant="assistant")
    answer = None
    for item in row["conversations"]:
        role = role_map.get(item["from"])
        if role == "assistant" and any(x["role"] == "user" for x in messages) and item["value"].strip():
            answer = item["value"]
            break
        if role in ("user", "system"):
            messages.append(dict(role=role, content=item["value"]))
    if answer is None:
        raise ValueError("sample lacks a user->assistant pair: " + row["id"])
    prompt = chat_ids(tokenizer, messages)
    continuation = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    rng = random.Random(SEED + row["row_index"])
    limit = min(len(continuation), 256)
    count = min(count, limit)
    offset = rng.randrange(max(1, limit - count + 1))
    before = prompt + continuation[:offset]
    before = before[-(context - count):]
    ids = before + continuation[offset:offset + count]
    return ids, list(range(len(before), len(ids))), continuation[offset + count] if offset + count < len(continuation) else -1


def load(args):
    parts, shards = [], []
    for rank in range(args.shards):
        path = args.capture / f"capture-{rank}.pt"
        shards.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        part = torch.load(path, map_location="cpu", weights_only=True)
        assert torch.equal(part["labels"], part["proposals"] == part["targets"][:, None])
        parts.append(part)
    hidden = torch.cat([p["hidden"] for p in parts])
    truth = torch.cat([p["labels"] for p in parts]).numpy().astype(bool)
    targets = torch.cat([p["targets"] for p in parts]).numpy()
    split, row, slot = [], [], []
    for part in parts:
        for s, r, c in zip(part["split"], part["row_ids"], part["counts"]):
            split += [s] * c
            row += [r] * c
            slot += list(range(c))
    del parts
    split = np.asarray(split)
    row = np.asarray(row)
    slot = np.asarray(slot)
    if not args.synthetic:
        assert hidden.shape == (95842, 5120) and truth.shape == (95842, 4)
        for s, (conversations, positions) in EXPECTED.items():
            assert len(set(row[split == s])) == conversations and int((split == s).sum()) == positions
    # Each window stores consecutive decode positions in order.
    prev = np.where(slot > 0, np.arange(len(slot)) - 1, -1)
    assert np.all(row[prev[slot > 0]] == row[slot > 0])
    rows = {}
    for s in SPLITS:
        with (args.data / f"{s}.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line)
                rows[record["id"]] = record
    source = np.asarray([rows[r]["source"] for r in row])
    meta = dict(split=split, row=row, slot=slot, prev=prev, source=source, targets=targets)
    if args.tokenizer:
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        token = np.full(len(row), -1)
        dataset_next = np.full(len(row), -1)
        start = np.flatnonzero(slot == 0)
        for begin, end in zip(start, np.r_[start[1:], len(row)]):
            ids, positions, after = window(tokenizer, rows[row[begin]])
            assert end - begin == len(positions) and np.all(row[begin:end] == row[begin]), row[begin]
            for j, p in enumerate(positions):
                token[begin + j] = ids[p]
                dataset_next[begin + j] = ids[p + 1] if p + 1 < len(ids) else after
        # On-policy proxy: the model's own argmax at t-1 equals the observed input token at t.
        agree = np.zeros(len(row), dtype=bool)
        agree[slot > 0] = targets[prev[slot > 0]] == token[slot > 0]
        meta.update(token=token, prev_agree=agree, target_matches_dataset=targets == dataset_next)
    return hidden, truth, meta, shards


def development_mask(meta):
    digest = np.asarray([int(hashlib.sha256(r.encode()).hexdigest()[:8], 16) % 10 == 0 for r in meta["row"]])
    return (meta["split"] == "train") & digest


# ---------------------------------------------------------------- models


def rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True).clamp_min(1e-12))


class Gate(nn.Module):
    def __init__(self, inputs, width, depth, outputs, dropout, norm):
        super().__init__()
        layers, d = [], inputs
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            if norm:
                layers.append(nn.LayerNorm(width))
            layers += [nn.GELU() if norm else nn.ReLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def features(hidden, prev, index, context):
    current = rms(hidden[index])
    if not context:
        return current
    return torch.cat((current, rms(hidden[prev[index]])), -1)


def macs(cfg, dim):
    d, total = dim * (2 if cfg["context"] else 1), 0
    for _ in range(cfg["depth"]):
        total += d * cfg["width"]
        d = cfg["width"]
    return total + d * (5 if cfg["loss"] == "ce5" else 4)


def longest_safe(truth):
    return np.where(truth, np.arange(1, 5), 0).max(-1)


def safety(cfg, logits):
    """Per-width safety score S[:, k] for action k+1."""
    if cfg["loss"] == "ce5":
        p = logits.softmax(-1)
        return p.flip(-1).cumsum(-1).flip(-1)[:, 1:], p
    return logits.sigmoid(), None


def loss_fn(cfg, logits, truth, target):
    if cfg["loss"] == "ce5":
        return F.cross_entropy(logits, target)
    y = truth.float()
    raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if cfg.get("focal"):
        p = logits.sigmoid()
        pt = torch.where(truth, p, 1 - p)
        raw = raw * (1 - pt).pow(cfg["focal"])
    weight = torch.where(truth, 1.0, cfg.get("negative_weight", 1.0))
    return (raw * weight).sum() / weight.sum()


@torch.inference_mode()
def predict(cfg, model, hidden, prev, index):
    model.eval()
    scores, probabilities = [], []
    for chunk in torch.as_tensor(index).split(4096):
        s, p = safety(cfg, model(features(hidden, prev, chunk.to(hidden.device), cfg["context"])))
        scores.append(s.float().cpu())
        if p is not None:
            probabilities.append(p.float().cpu())
    return torch.cat(scores).numpy(), (torch.cat(probabilities).numpy() if probabilities else None)


# ---------------------------------------------------------------- policies


def measure(choice, truth):
    accepted = choice > 0
    safe = truth[np.arange(len(choice)), np.maximum(choice - 1, 0)] & accepted
    tp, fp = int(safe.sum()), int((accepted & ~safe).sum())
    oracle = WIDTHS[longest_safe(truth)]
    saved = np.where(safe, WIDTHS[choice], 0)
    any_safe = int(truth.any(-1).sum())
    return dict(total=len(choice), accepted=int(accepted.sum()), tp=tp, fp=fp,
                precision=tp / (tp + fp) if tp + fp else None,
                coverage=float(accepted.mean()),
                recall=tp / any_safe if any_safe else None,
                mean_skipped_blocks=float(WIDTHS[choice].mean()),
                safe_skipped_blocks=float(saved.mean()),
                block_recall=float(saved.sum() / oracle.sum()) if oracle.sum() else None,
                action_counts={str(w): int((choice == i).sum()) for i, w in enumerate(ACTIONS)})


def choose(policy, scores, probabilities, value):
    if policy == "longest":
        eligible = scores >= value
        return np.where(eligible, np.arange(1, 5), 0).max(-1)
    if policy == "utility":
        # Expected saved blocks minus lambda times expected unsafe skips.
        utility = WIDTHS[1:] * scores - value * (1 - scores)
        best = utility.argmax(-1)
        return np.where(utility[np.arange(len(best)), best] > 0, best + 1, 0)
    if policy == "argmax":
        best = probabilities.argmax(-1)
        return np.where(probabilities.max(-1) >= value, best, 0)
    raise ValueError(policy)


def candidates(policy, scores, probabilities):
    if policy == "utility":
        return np.r_[0.0, np.logspace(-2, 6, 801)]
    values = probabilities.max(-1) if policy == "argmax" else scores.reshape(-1)
    return np.unique(np.r_[np.quantile(values, np.linspace(0, 1, 2001)), np.sort(values)[-400:], 1.000001])


def sweep(policy, scores, probabilities, truth):
    return [(float(v), measure(choose(policy, scores, probabilities, v), truth)) for v in candidates(policy, scores, probabilities)]


def calibrate(curve, goal):
    eligible = [(m["mean_skipped_blocks"], m["coverage"], value, m) for value, m in curve
                if m["accepted"] >= MINIMUM_ACCEPTED and m["precision"] >= goal]
    if not eligible:
        return None, None
    best = max(eligible, key=lambda x: x[:2])
    return best[2], best[3]


def bootstrap(choice, truth, rows, draws=1000):
    accepted = choice > 0
    safe = truth[np.arange(len(choice)), np.maximum(choice - 1, 0)] & accepted
    names, inverse = np.unique(rows, return_inverse=True)
    tp = np.bincount(inverse, safe, len(names))
    fp = np.bincount(inverse, accepted & ~safe, len(names))
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(names), size=(draws, len(names)))
    t, f = tp[sample].sum(1), fp[sample].sum(1)
    precision = t / np.maximum(t + f, 1)
    return [float(np.quantile(precision, 0.025)), float(np.quantile(precision, 0.975))]


def auroc(score, label):
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    positive = label.sum()
    negative = len(label) - positive
    if not positive or not negative:
        return None
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def evaluate(scores, probabilities, truth, meta, masks, policies):
    """Calibrate on one split, report on the other, both directions; plus the heldout frontier."""
    report = {}
    for policy in policies:
        curves = {s: sweep(policy, scores[masks[s]], None if probabilities is None else probabilities[masks[s]], truth[masks[s]])
                  for s in ("calibration", "heldout")}
        entry = {}
        for goal in GOALS:
            item = {}
            for fit_split, test_split in (("calibration", "heldout"), ("heldout", "calibration")):
                value, fitted = calibrate(curves[fit_split], goal)
                if value is None:
                    item[f"{fit_split}_to_{test_split}"] = None
                    continue
                choice = choose(policy, scores[masks[test_split]], None if probabilities is None else probabilities[masks[test_split]], value)
                result = measure(choice, truth[masks[test_split]])
                result["precision_ci95"] = bootstrap(choice, truth[masks[test_split]], meta["row"][masks[test_split]])
                item[f"{fit_split}_to_{test_split}"] = dict(value=value, fitted=fitted, test=result)
            _, frontier = calibrate(curves["heldout"], goal)
            item["heldout_frontier"] = frontier
            entry[str(goal)] = item
        report[policy] = entry
    held = masks["heldout"]
    report["heldout_auroc"] = {str(w): auroc(scores[held, k], truth[held, k]) for k, w in enumerate(SKIPS)}
    return report


def headline(report, goal="0.995"):
    # The policy family is chosen by its calibration-split fit, never by heldout.
    best = None
    for policy, entry in report.items():
        if policy == "heldout_auroc":
            continue
        item = entry[goal]["calibration_to_heldout"]
        if item and (best is None or item["fitted"]["mean_skipped_blocks"] > best[1]["fitted"]["mean_skipped_blocks"]):
            best = (policy, item)
    return None if best is None else dict(policy=best[0], value=best[1]["value"], **{k: best[1]["test"][k] for k in
                                          ("precision", "precision_ci95", "coverage", "recall", "mean_skipped_blocks", "block_recall")})


def development_score(cfg, scores, probabilities, truth):
    values = []
    curves = [sweep(policy, scores, probabilities, truth) for policy in ("longest", "utility")]
    for goal in (0.99, 0.995):
        best = 0.0
        for curve in curves:
            _, result = calibrate(curve, goal)
            if result:
                best = max(best, result["mean_skipped_blocks"])
        values.append(best)
    return float(np.mean(values)), values


# ---------------------------------------------------------------- training


def train(cfg, hidden, truth_gpu, target_gpu, prev_gpu, fit_index, dev_index, dev_truth, seed):
    torch.manual_seed(SEED + seed)
    np.random.seed(SEED + seed)
    dim = hidden.shape[1] * (2 if cfg["context"] else 1)
    model = Gate(dim, cfg["width"], cfg["depth"], 5 if cfg["loss"] == "ce5" else 4, cfg["dropout"], cfg["norm"]).to(hidden.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    fit = torch.as_tensor(fit_index, device=hidden.device)
    curves, best, step = [], None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total = 0.0
        for index in fit[torch.randperm(len(fit), device=fit.device)].split(1024):
            logits = model(features(hidden, prev_gpu, index, cfg["context"]))
            loss = loss_fn(cfg, logits, truth_gpu[index], target_gpu[index])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            step += 1
            total += float(loss.detach()) * len(index)
        scores, probabilities = predict(cfg, model, hidden, prev_gpu, dev_index)
        score, parts = development_score(cfg, scores, probabilities, dev_truth)
        curves.append(dict(epoch=epoch + 1, loss=total / len(fit), steps=step, development=score, development_by_goal=parts))
        if cfg["select"] == "last" or best is None or score > best[0]:
            best = (score, epoch + 1, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    model.eval()
    return model, curves, best[1], best[0]


CONFIGS = [
    dict(name="ce5_d1_fixed6", loss="ce5", depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=6, select="last", context=False),
    dict(name="ce5_d1", loss="ce5", depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=24, select="best", context=False),
    dict(name="bce4_d1", loss="bce4", depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=24, select="best", context=False),
    dict(name="bce4_d1_neg4", loss="bce4", negative_weight=4.0, depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=24, select="best", context=False),
    dict(name="bce4_d1_focal2", loss="bce4", focal=2.0, depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=24, select="best", context=False),
    dict(name="bce4_linear", loss="bce4", depth=0, width=0, norm=False, dropout=0.0, lr=3e-4, weight_decay=0.03, epochs=24, select="best", context=False),
    dict(name="bce4_d2", loss="bce4", depth=2, width=2048, norm=True, dropout=0.2, lr=2e-4, weight_decay=0.05, epochs=24, select="best", context=False),
    dict(name="ce5_d2", loss="ce5", depth=2, width=2048, norm=True, dropout=0.2, lr=2e-4, weight_decay=0.05, epochs=24, select="best", context=False),
    dict(name="ctx_bce4_d1", loss="bce4", depth=1, width=1024, norm=False, dropout=0.1, lr=1e-4, weight_decay=0.03, epochs=24, select="best", context=True),
    dict(name="ctx_bce4_d2", loss="bce4", depth=2, width=2048, norm=True, dropout=0.2, lr=2e-4, weight_decay=0.05, epochs=24, select="best", context=True),
    dict(name="ctx_ce5_d2", loss="ce5", depth=2, width=2048, norm=True, dropout=0.2, lr=2e-4, weight_decay=0.05, epochs=24, select="best", context=True),
]


def subgroup_report(choice, truth, meta, mask_index, keys):
    out = {}
    for key in keys:
        values = meta[key][mask_index]
        out[key] = {str(v): measure(choice[values == v], truth[mask_index][values == v]) for v in np.unique(values)}
    return out


def main(args):
    out = Path(os.environ.get("RUN_DIR", args.out or "."))
    (out / "models").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    tick = time.monotonic()
    hidden, truth, meta, shards = load(args)
    development = development_mask(meta)
    has_prev = meta["slot"] > 0
    masks = dict(calibration=(meta["split"] == "calibration") & has_prev, heldout=(meta["split"] == "heldout") & has_prev)
    masks_all = dict(calibration=meta["split"] == "calibration", heldout=meta["split"] == "heldout")
    fit_mask = (meta["split"] == "train") & ~development
    data_summary = dict(
        shards=shards, seconds=time.monotonic() - tick,
        positions={s: int((meta["split"] == s).sum()) for s in SPLITS},
        development_positions=int(development.sum()), development_conversations=len(set(meta["row"][development])),
        fit_positions=int(fit_mask.sum()), has_prev_positions={s: int(((meta["split"] == s) & has_prev).sum()) for s in SPLITS},
        label_rates={s: {str(w): float(truth[meta["split"] == s, k].mean()) for k, w in enumerate(SKIPS)} for s in SPLITS},
        longest_safe_counts={s: {str(w): int((longest_safe(truth[meta["split"] == s]) == i).sum()) for i, w in enumerate(ACTIONS)} for s in SPLITS},
        nonmonotone_fraction=float((truth[:, 3] & ~truth[:, 0]).mean()),
        oracle_mean_blocks={s: float(WIDTHS[longest_safe(truth[meta["split"] == s])].mean()) for s in SPLITS})
    if "prev_agree" in meta:
        agree = meta["prev_agree"]
        data_summary["on_policy_proxy"] = {
            s: {name: dict(positions=int(m.sum()), safe11=float(truth[m, 3].mean()), any_safe=float(truth[m].any(-1).mean()),
                           oracle_mean_blocks=float(WIDTHS[longest_safe(truth[m])].mean()))
                for name, m in (("prev_agree", (meta["split"] == s) & has_prev & agree),
                                ("prev_disagree", (meta["split"] == s) & has_prev & ~agree))}
            for s in SPLITS}
        data_summary["target_matches_dataset_rate"] = {s: float(meta["target_matches_dataset"][meta["split"] == s].mean()) for s in SPLITS}
    emit("DATA", **{k: v for k, v in data_summary.items() if k != "shards"})

    hidden_gpu = hidden.to(device)
    del hidden
    truth_gpu = torch.as_tensor(truth, device=device)
    target_gpu = torch.as_tensor(longest_safe(truth), device=device)
    prev_gpu = torch.as_tensor(np.maximum(meta["prev"], 0), device=device)
    all_index = np.arange(len(truth))
    dev_index = all_index[development & has_prev]
    results = {}

    # Deployed five-class classifier, unchanged, for the same evaluation.
    if args.deployed:
        assert args.synthetic or sha(args.deployed) == DEPLOYED_SHA
        saved = torch.load(args.deployed, map_location="cpu", weights_only=True)
        cfg = dict(name="deployed_five_class", loss="ce5", depth=1, width=1024, norm=False, dropout=0.1, context=False)
        model = Gate(hidden_gpu.shape[1], 1024, 1, 5, 0.1, False).to(device)
        model.load_state_dict(saved["state_dict"])
        scores, probabilities = predict(cfg, model, hidden_gpu, prev_gpu, all_index)
        results[cfg["name"]] = dict(config=cfg, macs=macs(cfg, hidden_gpu.shape[1]),
                                    has_prev=evaluate(scores, probabilities, truth, meta, masks, ("argmax", "longest", "utility")),
                                    all_positions=evaluate(scores, probabilities, truth, meta, masks_all, ("argmax", "longest", "utility")))
        results[cfg["name"]]["headline_has_prev"] = headline(results[cfg["name"]]["has_prev"])
        # Reproduce the recorded r1 cal99 policy exactly with its saved threshold.
        recorded = next(x for x in saved["policies"] if x["goal"] == 0.99)
        reproduced = {}
        for s in ("calibration", "heldout"):
            m = masks_all[s]
            reproduced[s] = measure(choose("argmax", scores[m], probabilities[m], recorded["threshold"]), truth[m])
            if not args.synthetic:
                for key in ("accepted", "tp", "fp"):
                    assert reproduced[s][key] == recorded[s][key], (s, key, reproduced[s], recorded[s])
        results[cfg["name"]]["recorded_cal99_reproduced"] = reproduced
        emit("DEPLOYED", headline=results[cfg["name"]]["headline_has_prev"], recorded_cal99=reproduced)
        np.savez_compressed(out / "scores-deployed_five_class.npz", scores=scores, probabilities=probabilities)

    configs = CONFIGS if not args.quick else [dict(c, epochs=min(c["epochs"], 2)) for c in CONFIGS[:3] + CONFIGS[8:9]]
    trained = {}
    for cfg in configs:
        start = time.monotonic()
        fit_index = all_index[fit_mask & (has_prev if cfg["context"] else True)]
        model, curves, epoch, dev = train(cfg, hidden_gpu, truth_gpu, target_gpu, prev_gpu, fit_index, dev_index, truth[dev_index], 0)
        scores, probabilities = predict(cfg, model, hidden_gpu, prev_gpu, all_index)
        policies = ("argmax", "longest", "utility") if cfg["loss"] == "ce5" else ("longest", "utility")
        entry = dict(config=cfg, macs=macs(cfg, hidden_gpu.shape[1]), parameters=sum(p.numel() for p in model.parameters()),
                     curves=curves, selected_epoch=epoch, development=dev, seconds=time.monotonic() - start,
                     has_prev=evaluate(scores, probabilities, truth, meta, masks, policies))
        if not cfg["context"]:
            entry["all_positions"] = evaluate(scores, probabilities, truth, meta, masks_all, policies)
        entry["headline_has_prev"] = headline(entry["has_prev"])
        results[cfg["name"]] = entry
        trained[cfg["name"]] = (cfg, scores, probabilities)
        torch.save(dict(config=cfg, state_dict=model.cpu().state_dict(), seed=0, epoch=epoch), out / "models" / f"{cfg['name']}-seed0.pt")
        emit("CONFIG", name=cfg["name"], development=dev, epoch=epoch, seconds=entry["seconds"], headline=entry["headline_has_prev"])

    # Ensembles of the best plain and best context configuration, chosen by development only.
    ensembles = {}
    for family in (False, True):
        names = [n for n, (c, _, _) in trained.items() if c["context"] == family and c["select"] == "best"]
        if not names:
            continue
        name = max(names, key=lambda n: results[n]["development"])
        cfg = trained[name][0]
        seeds_scores, seeds_probabilities = [trained[name][1]], [trained[name][2]]
        dev_values = []
        for seed in range(1, args.ensemble):
            fit_index = all_index[fit_mask & (has_prev if cfg["context"] else True)]
            model, _, epoch, dev = train(cfg, hidden_gpu, truth_gpu, target_gpu, prev_gpu, fit_index, dev_index, truth[dev_index], seed)
            s, p = predict(cfg, model, hidden_gpu, prev_gpu, all_index)
            seeds_scores.append(s)
            seeds_probabilities.append(p)
            dev_values.append(dict(seed=seed, epoch=epoch, development=dev))
            torch.save(dict(config=cfg, state_dict=model.cpu().state_dict(), seed=seed, epoch=epoch), out / "models" / f"{name}-seed{seed}.pt")
        scores = np.mean(seeds_scores, 0)
        probabilities = None if seeds_probabilities[0] is None else np.mean(seeds_probabilities, 0)
        policies = ("argmax", "longest", "utility") if cfg["loss"] == "ce5" else ("longest", "utility")
        ensemble_dev, _ = development_score(cfg, scores[dev_index], None if probabilities is None else probabilities[dev_index], truth[dev_index])
        key = f"ensemble{args.ensemble}_{name}"
        entry = dict(member=name, seeds=args.ensemble, members=dev_values, development=ensemble_dev,
                     macs=args.ensemble * macs(cfg, hidden_gpu.shape[1]),
                     has_prev=evaluate(scores, probabilities, truth, meta, masks, policies))
        if not cfg["context"]:
            entry["all_positions"] = evaluate(scores, probabilities, truth, meta, masks_all, policies)
        entry["headline_has_prev"] = headline(entry["has_prev"])
        results[key] = entry
        ensembles[key] = (cfg, scores, probabilities)
        np.savez_compressed(out / f"scores-{key}.npz", scores=scores, **({} if probabilities is None else dict(probabilities=probabilities)))
        emit("ENSEMBLE", name=key, development=ensemble_dev, headline=entry["headline_has_prev"])

    # Recommendation by development score only; heldout numbers are reported, never used to choose.
    candidates_all = {n: results[n]["development"] for n in results if "development" in results[n]}
    recommended = max(candidates_all, key=candidates_all.get)
    cfg, scores, probabilities = (ensembles.get(recommended) or trained[recommended])
    best_policy = headline(results[recommended]["has_prev"])
    held = np.flatnonzero(masks["heldout"])
    breakdown = {}
    if best_policy:
        choice = choose(best_policy["policy"], scores[held], None if probabilities is None else probabilities[held], best_policy["value"])
        keys = ["source"] + (["prev_agree"] if "prev_agree" in meta else [])
        breakdown = subgroup_report(choice, truth, meta, held, keys)
        if "deployed_five_class" in results:
            deployed = np.load(out / "scores-deployed_five_class.npz")
            d_best = headline(results["deployed_five_class"]["has_prev"])
        if "deployed_five_class" in results and d_best:
            d_choice = choose(d_best["policy"], deployed["scores"][held], deployed["probabilities"][held], d_best["value"])
            breakdown["deployed_same_goal"] = subgroup_report(d_choice, truth, meta, held, keys)
    summary = dict(data=data_summary, results=results, recommended=recommended, recommended_headline=best_policy,
                   subgroups_heldout_goal_0995=breakdown,
                   protocol=dict(seed=SEED, goals=GOALS, minimum_accepted=MINIMUM_ACCEPTED,
                                 selection="Configuration and ensemble member by development (10% of training conversations, has_prev positions) mean of best mean-skipped-blocks at precision 0.99 and 0.995.",
                                 thresholds="Policy parameter chosen on calibration split; reported on heldout; swapped direction also reported. heldout_frontier is the heldout-optimal parameter, for ranking only.",
                                 evaluation_subset="has_prev = window slots 1..7 (previous decode position captured) for all models; all_positions additionally for single-token models.",
                                 label="Native cached one-step counterfactual argmax agreement at each width, teacher-forced dataset continuation."),
                   runtime=dict(torch=str(torch.__version__), device=device,
                                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None),
                   seconds=time.monotonic() - tick)
    write(out / "summary.json", summary)
    emit("PASS", recommended=recommended, headline=best_policy, seconds=summary["seconds"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--deployed", type=Path)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--out", type=Path)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--quick", action="store_true")
    main(p.parse_args())
