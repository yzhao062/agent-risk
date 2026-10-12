"""Arm E: does a shared seed give two models the same sampling noise?

vLLM 0.30.0 samples a seeded request by the exponential race
(v1/sample/ops/topk_topp_sampler.py: random_sample): a per-request
torch.Generator seeded with the request seed fills that request's row of a float32
(batch, logits width) tensor with exponential noise, and the token is
argmax(probs / noise), a Gumbel-max draw. The logits width is the model's
config vocab_size: 151,936 for Qwen3-14B (FP8 and bf16), 152,064 for
Qwen2.5-14B-Instruct, and 248,320 for Qwen3.5-27B. This script reproduces that call
on the GPU for each width and reports whether the noise on the shared token ids
(the first 151,936) is identical across widths, over several decode steps and for
the seeded row at different batch positions. Run inside the vLLM container:

  apptainer exec --nv vllm.sif python3 noise_check.py
"""

import json

import torch

WIDTHS = [151936, 152064, 248320]
SHARED = 151936
SEEDS = [200000000, 200042007, 200099014]
STEPS = 8


def noise(width: int, seed: int, row: int, batch: int) -> torch.Tensor:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.empty((batch, width), dtype=torch.float32, device="cuda")
    out = []
    for _ in range(STEPS):
        q.exponential_()  # unseeded rows, as vLLM does when not every row has a seed
        q[row].exponential_(generator=g)
        out.append(q[row, :SHARED].clone())
    return torch.stack(out)


def main() -> None:
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "steps": STEPS, "cases": []}
    for seed in SEEDS:
        base = noise(WIDTHS[0], seed, row=0, batch=1)
        case = {"seed": seed,
                "same_width_other_batch_position": bool(torch.equal(base, noise(WIDTHS[0], seed, row=3, batch=4)))}
        for w in WIDTHS[1:]:
            other = noise(w, seed, row=0, batch=1)
            case[f"width_{w}_identical_on_shared_ids"] = bool(torch.equal(base, other))
            case[f"width_{w}_fraction_equal"] = round(float((base == other).float().mean()), 6)
        res["cases"].append(case)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
