import torch
import torch.distributed as dist
from torch import Tensor

POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step(p: Tensor, g: Tensor, m: Tensor, v: Tensor, step: Tensor, lr: Tensor, b1: Tensor, b2: Tensor, eps: Tensor, wd: Tensor):
    p.mul_(1 - lr * wd)
    m.lerp_(g, 1 - b1)
    v.lerp_(g.square(), 1 - b2)
    bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
    p.add_(m / bc1 / ((v / bc2).sqrt() + eps), alpha=-lr)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step(g: Tensor, p: Tensor, m: Tensor, v: Tensor, mom: Tensor, lr: Tensor, wd: Tensor, b2: Tensor, ns: int, rd: int):
    mom_f = mom.to(g.dtype)
    m.lerp_(g, 1 - mom_f)
    g = g.lerp_(m, mom_f)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_COEFFS[:ns]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in POLAR_COEFFS[:ns]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    b2_f = b2.to(g.dtype)
    v_mean = g.float().square().mean(dim=rd, keepdim=True)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * g.size(rd)).sqrt()
    v.lerp_(v_mean.to(v.dtype), (1 - b2_f).to(v.dtype))
    scale = v.clamp_min(1e-10).rsqrt()
    v_new = ((v_mean * g.size(rd)) * scale.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (scale * (v_norm / v_new.clamp_min(1e-10))).to(g.dtype)
    lr_f, wd_f = lr.to(g.dtype), wd.to(g.dtype)
    mask = (g * p) >= 0
    p.sub_(lr_f * g + lr_f * wd_f * p * mask)

class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._t = {k: torch.tensor(0.0, dtype=torch.float32) for k in ['step', 'lr', 'b1', 'b2', 'eps', 'wd', 'mom', 'muon_lr', 'muon_wd', 'muon_b2']}

    def _adamw(self, group: dict):
        for p in group['params']:
            if p.grad is None:
                continue
            s = self.state[p]
            if not s:
                s['step'], s['m'], s['v'] = 0, torch.zeros_like(p), torch.zeros_like(p)
            s['step'] += 1
            self._t['step'].fill_(s['step'])
            self._t['lr'].fill_(group['lr'])
            self._t['b1'].fill_(group['betas'][0])
            self._t['b2'].fill_(group['betas'][1])
            self._t['eps'].fill_(group['eps'])
            self._t['wd'].fill_(group['weight_decay'])
            adamw_step(p, p.grad, s['m'], s['v'], self._t['step'], self._t['lr'], self._t['b1'], self._t['b2'], self._t['eps'], self._t['wd'])

    def _muon(self, group: dict):
        params = group['params']
        if not params:
            return
        p0 = params[0]
        s = self.state[p0]
        shape = p0.shape
        rd = -1 if shape[-2] >= shape[-1] else -2
        if 'm' not in s:
            s['m'] = torch.zeros(len(params), *shape, dtype=p0.dtype, device=p0.device)
            vs = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            s['v'] = torch.zeros(vs, dtype=p0.dtype, device=p0.device)
        # Handle None gradients (can happen with set_to_none=True) by treating as zeros
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        sg = torch.stack(grads)
        sp = torch.stack(params)
        self._t['mom'].fill_(group['momentum'])
        self._t['muon_b2'].fill_(group.get('beta2', 0.95))
        self._t['muon_lr'].fill_(group['lr'] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._t['muon_wd'].fill_(group['weight_decay'])
        muon_step(sg, sp, s['m'], s['v'], self._t['mom'], self._t['muon_lr'], self._t['muon_wd'], self._t['muon_b2'], group.get('ns_steps', 5), rd)
        torch._foreach_copy_(params, list(sp.unbind(0)))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if g['kind'] == 'adamw':
                self._adamw(g)
            elif g['kind'] == 'muon':
                self._muon(g)

def get_muon_param_groups(named_params, adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.0, muon_lr=0.02, muon_momentum=0.95, muon_wd=0.2, muon_ns_steps=5, verbose=False):
    adamw_params, muon_groups = [], {}
    for name, p in named_params:
        if not p.requires_grad:
            continue
        ln = name.lower()
        is_embed = any(x in ln for x in ['wte', 'embedding', 'embed', 'lm_head'])
        is_norm = any(x in ln for x in ['norm', 'ln_', 'bias'])
        # Parcae recurrent params: A_log, B, C, dt_bias - use AdamW for stability
        is_parcae_recurrent = any(x in ln for x in ['a_log', 'dt_bias', 'adapter.b', 'adapter.adapter', 'transformer.c.'])
        if p.ndim < 2 or is_embed or is_norm or is_parcae_recurrent:
            adamw_params.append(p)
            if verbose:
                print(f"AdamW: {name}")
        else:
            key = p.shape
            if key not in muon_groups:
                muon_groups[key] = []
            muon_groups[key].append(p)
            if verbose:
                print(f"Muon: {name} {p.shape}")
    groups = [{'params': adamw_params, 'kind': 'adamw', 'lr': adamw_lr, 'base_lr': 1.0, 'betas': adamw_betas, 'eps': adamw_eps, 'weight_decay': adamw_wd}]
    for shape, params in muon_groups.items():
        groups.append({'params': params, 'kind': 'muon', 'lr': muon_lr, 'base_lr': muon_lr / adamw_lr, 'momentum': muon_momentum, 'weight_decay': muon_wd, 'ns_steps': muon_ns_steps})
    return groups

