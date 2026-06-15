"""qat_engine.py — 6/19 Sprint 단일 진실원 (S0 부분).

Budget-Dependent Quantization Recovery · CNN·W4 cell.
매핑: 연구계획서 §4(동역학) · 실험계획서 §2-5,§8-9 · sprint 05/06.

이 파일에 박힌 "잠긴 결정 + 합의된 수정"
--------------------------------------------------------------------
[엔진]   manual additive-STE fake-quant, per-channel, scale은 calib로 한 번 정해 *고정*.   (04 §8, 06 §①, #4)
[mom=0]  진단/핵심 run은 vanilla SGD(momentum=0). §4.2 닫힌형태는 vanilla GD((1-ηλ)^t)이고,
         heavy-ball은 저곡률 방향을 가속해 방향별 수렴속도 격차(=단기 λ² 가중이 사는 곳)를 압축한다.
         → 0.9는 11월 "가속에도 살아남나" 축. (기존 lock SGD mom0.9 → 0으로 교정.)             (#3)
[silent] PTQ 후 (a)양자화된 층 수==기대, (b)W가 FP32와 *실제로* 다름 을 assert.
         torchao가 conv를 말없이 0개 양자화했던 그 사고를 manual에서도 막는다.                  (#1 claude.ai)
[W8san]  W8 sanity = (거의 무손실) + (round-trip 단위테스트). conv는 torchao로 검증 불가
         (P0B: torchao가 resnet18 conv 0개 양자화) → manual==torchao는 fc에서만 보조.            (#2)
[noise]  노이즈 바닥 = "단일층 *recovery*"의 run-to-run std (고정모델 eval 분산 아님).            (#7)
[device] swap 후 새 모듈을 원본 device로 이동 (파일럿 device-order 버그 방지).
[BNfrz]  set_trainable에서 BN running stats 고정(eval) — 통계 갱신이 숨은 학습 되지 않게.        (04 §5-7)

S1에서 추가 예정(여기 없음): proxy_scores(5종 sweep) · select_subset · spearman · inversion_strength.
"""

import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
# 0. 재현성
# =====================================================================
def set_seed(seed: int):
    """seed 고정. 단, cudnn.deterministic은 *일부러* 끈다 —
    run-to-run 변동(=노이즈 바닥)이 우리가 측정하려는 대상이기 때문."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# =====================================================================
# 1. 모델 & 데이터
# =====================================================================
def load_model(name='resnet18', dataset='cifar100', ckpt=None, device=DEVICE):
    """CIFAR용 ResNet-18.
    ⚠ timm pretrained=True는 ImageNet(224·1000·7x7 stem)이라 CIFAR에 그대로 못 씀.
      → pretrained=False로 만들고 stem을 32x32용으로 수술(3x3 stride1, maxpool 제거),
        FP32 baseline은 train_baseline()으로 *한 번* 학습→캐시하고 ckpt로 로드.        (#1)
    ckpt 경로 주면 그걸 로드(= 백업본 swap, 한 줄)."""
    import timm
    num_classes = {'cifar100': 100, 'cifar10': 10}[dataset]
    model = timm.create_model(name, pretrained=False, num_classes=num_classes)
    if dataset.startswith('cifar') and name == 'resnet18':
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if ckpt is not None:
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd)
    return model.to(device)


_CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
_CIFAR100_STD  = (0.2673, 0.2564, 0.2762)


def get_loaders(dataset='cifar100', batch=128, calib_size=512,
                data_root='./data', num_workers=2):
    """train(증강O) / val / calib(증강X, train에서 calib_size장) 로더.
    calib = PTQ scale + (S1)HVP/Fisher 추정 공용 — 증강 없이 고정."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    assert dataset == 'cifar100', "S0는 cifar100 고정"
    norm = transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_set = datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tf)
    val_set   = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    calib_src = datasets.CIFAR100(data_root, train=True,  download=True, transform=test_tf)
    calib_set = Subset(calib_src, list(range(calib_size)))

    train = DataLoader(train_set, batch_size=batch, shuffle=True,  num_workers=num_workers, drop_last=True)
    val   = DataLoader(val_set,   batch_size=256,   shuffle=False, num_workers=num_workers)
    calib = DataLoader(calib_set, batch_size=batch, shuffle=False, num_workers=num_workers)
    return train, val, calib


def train_baseline(model, train_loader, val_loader, epochs=60, lr=0.1, momentum=0.9,
                   wd=5e-4, ckpt_path='checkpoints/resnet18_cifar100_fp32.pt',
                   device=DEVICE, resume=True):
    """제대로 된 FP32 CIFAR-100 baseline(~76%) 한 번 학습 → 캐시. (#2 claude.ai · 이론 위생)
    이미 ckpt가 있으면 로드만(재학습 0). 이 baseline 학습 비용은 *일회성*이지 iteration 비용이 아님."""
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    if resume and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        acc = evaluate(model.to(device), val_loader, device)
        print(f"[baseline] 캐시 로드 {ckpt_path} → top1 {acc:.2f}")
        return model, acc

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.0)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, val_loader, device)
        if acc > best:
            best = acc; torch.save(model.state_dict(), ckpt_path)
        print(f"[baseline] epoch {ep+1}/{epochs}  top1 {acc:.2f}  best {best:.2f}")
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.to(device), best


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    """Top-1(%)."""
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total


# =====================================================================
# 2. 양자화 — manual additive-STE, per-channel, *고정* scale
# =====================================================================
def fake_quant(w, scale, n_bits):
    """대칭 per-channel fake-quant + additive STE.  Q(w) = clamp(round(w/s), -qmax, qmax)*s
    반환값은 w + (wq - w).detach() → 값은 wq, gradient는 w로 직통(STE).
    scale은 *인자로 받는다*(매 forward 재계산 X) → 양자화 격자 고정 → δ·H가 학습 내내 well-defined. (#4)"""
    qmax = 2 ** (n_bits - 1) - 1
    wq = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (wq - w).detach()


def compute_scales(model, n_bits, per_channel=True):
    """각 Conv2d/Linear 가중치에서 per-channel amax 기반 scale을 *한 번* 계산해 dict로.
    weight-only 대칭 양자화라 calib 데이터 불필요(가중치 자체에서 결정). 활성화 양자화는 11월."""
    qmax = 2 ** (n_bits - 1) - 1
    scales = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            if per_channel:
                dims = list(range(1, w.ndim))
                amax = w.abs().amax(dim=dims, keepdim=True)
            else:
                amax = w.abs().amax()
            scales[name] = (amax / qmax).clamp_min(1e-12)
    return scales


class QConv2d(nn.Conv2d):
    """forward에서 가중치에 고정-scale fake-quant 적용."""
    @classmethod
    def from_float(cls, m: nn.Conv2d, scale, n_bits):
        q = cls(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                padding=m.padding, dilation=m.dilation, groups=m.groups,
                bias=m.bias is not None)
        q.weight = nn.Parameter(m.weight.detach().clone())
        if m.bias is not None:
            q.bias = nn.Parameter(m.bias.detach().clone())
        q.register_buffer('scale', scale.clone())
        q.n_bits = n_bits
        return q

    def forward(self, x):
        wq = fake_quant(self.weight, self.scale, self.n_bits)
        return self._conv_forward(x, wq, self.bias)


class QLinear(nn.Linear):
    @classmethod
    def from_float(cls, m: nn.Linear, scale, n_bits):
        q = cls(m.in_features, m.out_features, bias=m.bias is not None)
        q.weight = nn.Parameter(m.weight.detach().clone())
        if m.bias is not None:
            q.bias = nn.Parameter(m.bias.detach().clone())
        q.register_buffer('scale', scale.clone())
        q.n_bits = n_bits
        return q

    def forward(self, x):
        wq = fake_quant(self.weight, self.scale, self.n_bits)
        return F.linear(x, wq, self.bias)


def _set_submodule(model, name, new_module):
    """'layer1.0.conv1' 같은 점-경로로 서브모듈 교체."""
    parts = name.split('.'); parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def ptq(model, n_bits, calib=None, tol=1e-8, device=DEVICE):
    """Conv2d/Linear → QConv2d/QLinear 로 swap, 고정 scale fake-quant 적용.
    silent-skip 가드 2종 박음:
      (a) 교체된 층 수 == 기대(Conv2d+Linear 총수)
      (b) 모든 대상 층에서 fake-quant가 가중치를 *실제로* 바꿈
    하나라도 깨지면 = wiring 버그(말없이 양자화 누락)를 밤1에 즉시 잡는다. (#1)"""
    scales = compute_scales(model, n_bits)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    expected = len(targets)
    fp_weights = {n: m.weight.detach().clone() for n, m in targets}

    replaced = 0
    for name, m in targets:
        q = (QConv2d if isinstance(m, nn.Conv2d) else QLinear).from_float(m, scales[name], n_bits)
        _set_submodule(model, name, q.to(device))      # ← device-order 버그 방지
        replaced += 1
    assert replaced == expected, f"[silent-skip] 교체 {replaced} != 기대 {expected}"

    changed = 0
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            with torch.no_grad():
                wq = fake_quant(m.weight, m.scale, n_bits)
            if (wq - fp_weights[name].to(wq.device)).abs().max().item() > tol:
                changed += 1
    assert changed == expected, f"[silent-skip] 실제 양자화 {changed}/{expected} 층 — 일부가 안 걸림"
    return model.to(device)


def quant_error(model):
    """δ_l = W_l - Q(W_l)  (양자화 오차, §4.4). 학습 전 PTQ 모델 위에서 호출."""
    deltas = {}
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QLinear)):
            with torch.no_grad():
                qmax = 2 ** (m.n_bits - 1) - 1
                wq = torch.clamp(torch.round(m.weight / m.scale), -qmax, qmax) * m.scale
                deltas[name] = (m.weight.detach() - wq)
    return deltas


def roundtrip_test(n_bits=8, device=DEVICE):
    """W8 sanity의 conv 경로 검증: 알려진 텐서 round-trip.
    (1) 양자화가 텐서를 실제로 바꾸나, (2) 오차가 이론한계 scale/2 이내인가."""
    w = torch.randn(64, 32, 3, 3, device=device)
    qmax = 2 ** (n_bits - 1) - 1
    dims = list(range(1, w.ndim))
    scale = (w.abs().amax(dim=dims, keepdim=True) / qmax).clamp_min(1e-12)
    wq = fake_quant(w, scale, n_bits)
    assert not torch.equal(wq, w), "양자화가 텐서를 안 바꿈"
    max_err = (wq - w).abs().max().item()
    bound = scale.max().item() / 2 * (1 + 1e-4)
    assert max_err <= bound, f"양자화 오차 {max_err:.3e} > 이론한계 {bound:.3e}"
    return dict(n_bits=n_bits, max_err=max_err, bound=bound, ok=True)


# =====================================================================
# 3. partial QAT — 단일층/subset freeze + 짧은 회복 루프
# =====================================================================
def _freeze_bn_stats(model):
    """BN을 eval로 → running stats 갱신 중단(숨은 학습 차단). 모델이 train()이어도 BN만 고정."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def set_trainable(model, layer_names):
    """layer_names의 weight만 grad ON, 나머지 전부 OFF + BN stats 고정.
    layer_names에 여러 층을 주면 block/subset 단위(= 신호 약할 때 거칠게 가는 손잡이, #10)."""
    for p in model.parameters():
        p.requires_grad_(False)
    name_set = set(layer_names)
    hit = 0
    for name, m in model.named_modules():
        if name in name_set:
            for p in m.parameters(recurse=False):
                p.requires_grad_(True); hit += 1
    assert hit > 0, f"set_trainable: {layer_names} 중 매칭된 학습 파라미터 0개 — 이름 확인"
    _freeze_bn_stats(model)
    return model


def short_qat(model, train_loader, val_loader, steps=None, lr=1e-3, momentum=0.0,
              seed=0, eval_at=(30, 100, 300), plateau=False,
              plateau_every=100, plateau_patience=5, plateau_eps=0.1,
              max_plateau_steps=5000, device=DEVICE, return_state=False):
    """짧은 QAT 회복 루프. 한 *궤적*을 돌며 eval_at 체크포인트에서 top1 기록
    → {t: top1} 반환(여러 run이 아니라 한 run의 체크포인트, 04 §2.6/§5-2).
    momentum=0 기본(#3). plateau=True면 더 안 오를 때까지 돌고 'plateau' 키 추가(수렴 대용).
    return_state=True면 plateau 가중치(=단일층 실험의 경험적 φ*) state_dict도 반환 → δ_true vs δ_approx 공짜검증(#6)."""
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "학습 파라미터 0 — set_trainable 먼저"
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    model.train(); _freeze_bn_stats(model)
    eval_at = sorted(set(int(t) for t in eval_at))
    results, state = {}, None
    best, since_improve = -1.0, 0

    def _do_eval(tag):
        nonlocal results
        results[tag] = evaluate(model, val_loader, device)
        model.train(); _freeze_bn_stats(model)

    hard_cap = steps or (max_plateau_steps if plateau else max(eval_at))
    step = 0; it = iter(train_loader)
    while step < hard_cap:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        step += 1

        if step in eval_at:
            _do_eval(step)
        if plateau and step % plateau_every == 0:
            cur = evaluate(model, val_loader, device); model.train(); _freeze_bn_stats(model)
            if cur > best + plateau_eps:
                best = cur; since_improve = 0
            else:
                since_improve += 1
                if since_improve >= plateau_patience:
                    results['plateau'] = cur; break
    if plateau and 'plateau' not in results:
        results['plateau'] = evaluate(model, val_loader, device)
    if return_state:
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return results, state
    return results


# =====================================================================
# 4. HVP (Pearlmutter) — 검증됨(additive-STE 위 autograd HVP, float64 rel 3.8e-7)
# =====================================================================
def hvp_layer(model, layer_name, delta, x, y, criterion=None, device=DEVICE):
    """층 l의 H_l·δ_l 한 번 (추가 backward 1회). 04 §3 코드 그대로."""
    criterion = criterion or nn.CrossEntropyLoss()
    module = dict(model.named_modules())[layer_name]
    W = module.weight
    prev = W.requires_grad; W.requires_grad_(True)
    model.eval()                                  # 곡률은 결정적 forward에서(BN/dropout 고정)
    x, y, delta = x.to(device), y.to(device), delta.to(device)
    loss = criterion(model(x), y)
    g  = torch.autograd.grad(loss, W, create_graph=True)[0]      # ∂L/∂W
    Hv = torch.autograd.grad((g * delta).sum(), W)[0]            # H·δ
    W.requires_grad_(prev)
    return Hv.detach()


def hvp_proxies(model, layer_name, delta, calib_loader, n_batches=4, device=DEVICE):
    """calib 여러 배치로 E[H]δ 추정 후 두 proxy.
      V_converge = δᵀHδ (수렴) , V_short = ‖Hδ‖² (단기) , 그리고 δᵀHδ의 *부호*(#5).
    부호 음수 = PTQ점의 진짜 H가 PSD 아님(§4.1 H⪰0은 이상화) → 버그 아니라 breakdown 신호/GGN 힌트."""
    crit = nn.CrossEntropyLoss()
    Hd_sum, n = None, 0
    for i, (x, y) in enumerate(calib_loader):
        if i >= n_batches:
            break
        Hd = hvp_layer(model, layer_name, delta, x, y, crit, device)
        Hd_sum = Hd if Hd_sum is None else Hd_sum + Hd
        n += 1
    Hd = Hd_sum / max(n, 1)
    dtHd = (delta.to(Hd.device) * Hd).sum().item()
    return dict(dtHd=dtHd, normHd2=(Hd * Hd).sum().item(),
                sign=int(np.sign(dtHd)), n_batches=n)


# =====================================================================
# 5. 노이즈 바닥 & 로깅
# =====================================================================
def noise_floor(run_fn, n=5):
    """run_fn(seed)->scalar 를 n번 → (mean, std).
    ⚠ run_fn은 '대표 단일층을 짧게 회복시키고 recovery(%p)를 반환'하는 클로저여야 함
      (고정모델 eval 분산이 아니라 *회복 측정 전체*의 run-to-run 변동). (#7)"""
    vals = [float(run_fn(seed=i)) for i in range(n)]
    return float(np.mean(vals)), float(np.std(vals)), vals


def log_run(config: dict, results: dict, path='outputs/runs.jsonl'):
    """run 하나 = config(모델·비트·층·t·B·seed…) + results 를 jsonl 한 줄로.
    분석은 이 파일 하나에서 (04 §4). 키가 run마다 달라도 jsonl이 안전."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({**config, 'results': results}, ensure_ascii=False) + '\n')


def load_runs(path='outputs/runs.jsonl'):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]
