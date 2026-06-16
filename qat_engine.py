"""qat_engine.py — 6/19 Sprint 단일 진실원 (S0 부분).

Budget-Dependent Quantization Recovery · CNN·W4 cell.
매핑: 연구계획서 §4(동역학) · 실험계획서 §2-5,§8-9 · sprint 05/06.

이 파일에 박힌 "잠긴 결정 + 합의된 수정"
--------------------------------------------------------------------
[엔진]   manual additive-STE fake-quant, per-channel, scale은 가중치에서 한 번 정해 *고정*(weight-only, activation calib 아님).   (04 §8, 06 §①, #4)
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

S1 구현됨(§6): proxy_scores(5종) · select_subset · spearman · inversion_strength · fisher_diag · isolated_output_delta · make_ptq_model 등.
"""

import os, json, random, copy
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
    calib = (S1)HVP/Fisher 추정용 — weight-only PTQ scale엔 calib 안 씀(가중치서 결정). 증강 없이 고정."""
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


@torch.no_grad()
def evaluate_full(model, loader, device=DEVICE, criterion=None):
    """Top-1(%) + 평균 val loss를 한 번에 (loss-R 측정용 — proxy=loss곡률과 같은 공간서 회복 비교, S1.2)."""
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval(); correct = total = 0; loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total, loss_sum / total


# =====================================================================
# 2. 양자화 — manual additive-STE, per-channel, *고정* scale
# =====================================================================
def fake_quant(w, scale, n_bits):
    """대칭 per-channel fake-quant + additive STE.  Q(w) = clamp(round(w/s), -qmax, qmax)*s
    반환값은 w + (wq - w).detach() → 값은 wq, gradient는 w로 직통(STE).
    scale은 *인자로 받는다*(매 forward 재계산 X) → 양자화 격자 고정 → δ·H가 학습 내내 well-defined. (#4)"""
    assert n_bits >= 2, "manual 엔진은 W2+ 전용 (W1은 qmax=0 → GSB로 처리, #5 Codex)"
    qmax = 2 ** (n_bits - 1) - 1
    wq = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
    return w + (wq - w).detach()


def compute_scales(model, n_bits, per_channel=True):
    """각 Conv2d/Linear 가중치에서 per-channel amax 기반 scale을 *한 번* 계산해 dict로.
    weight-only 대칭 양자화라 calib 데이터 불필요(가중치 자체에서 결정). 활성화 양자화는 11월."""
    assert n_bits >= 2, "manual 엔진은 W2+ 전용 (W1은 qmax=0 → GSB로 처리, #5 Codex)"
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
            w = getattr(m, 'weight', None)   # weight-only: bias는 열지 않음(통제 유지, Codex)
            if w is not None:
                w.requires_grad_(True); hit += 1
    assert hit > 0, f"set_trainable: {layer_names} 중 매칭된 학습 파라미터 0개 — 이름 확인"
    _freeze_bn_stats(model)
    return model


def short_qat(model, train_loader, val_loader, steps=None, lr=1e-3, momentum=0.0,
              seed=0, eval_at=(30, 100, 300), plateau=False,
              plateau_every=100, plateau_patience=5, plateau_eps=0.1,
              max_plateau_steps=5000, device=DEVICE, return_state=False,
              track_loss=False):
    """짧은 QAT 회복 루프. 한 *궤적*을 돌며 eval_at 체크포인트에서 top1(+옵션 loss) 기록
    → {t: top1} 반환 (track_loss=True면 {t: {'acc','loss'}}). 한 run의 체크포인트(04 §2.6/§5-2).
    momentum=0 기본(#3). plateau=True=적응형 조기종료(층마다 다른 예산·느린 마라톤층 절단 위험, eps 클수록 심함);
    S1.2는 plateau=False + steps=고정예산 + eval_at=고정 grid 권장(공정 비교·마라톤층 절단 제거).
    track_loss=True → loss-R도 기록(proxy=loss곡률과 같은 공간서 회복 검증 = apples-to-apples).
    return_state=True면 가중치 state_dict 반환(=단일층 실험의 경험적 φ*) → δ_true vs δ_approx 공짜검증(#6)."""
    set_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "학습 파라미터 0 — set_trainable 먼저"
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    model.train(); _freeze_bn_stats(model)
    eval_at = sorted(set(int(t) for t in eval_at))
    if steps is not None and not plateau:
        assert max(eval_at) <= steps, f"eval_at 최대({max(eval_at)}) > steps({steps}) — 뒷 시점이 조용히 빠짐 (Codex)"
    results, state = {}, None
    best, since_improve = -1.0, 0

    def _do_eval(tag):
        nonlocal results
        if track_loss:
            a, l = evaluate_full(model, val_loader, device)
            results[tag] = {'acc': a, 'loss': l}
        else:
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
                    _do_eval('plateau'); break
    if plateau and 'plateau' not in results:
        _do_eval('plateau')
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
    try:                                              # 에러나도 requires_grad 원복 (Codex)
        loss = criterion(model(x), y)
        g  = torch.autograd.grad(loss, W, create_graph=True)[0]      # ∂L/∂W
        Hv = torch.autograd.grad((g * delta).sum(), W)[0]            # H·δ
    finally:
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


# =====================================================================
# 6. S1 — proxy sweep + 순위분석 (역전·예측)
# =====================================================================
def list_quant_layers(model):
    """양자화 층 이름 목록 (QConv2d/QLinear)."""
    return [n for n, m in model.named_modules() if isinstance(m, (QConv2d, QLinear))]


def get_layer_costs(model, layers=None):
    """층별 학습 비용 ∝ 파라미터 수 (S2 예산 B용)."""
    layers = layers or list_quant_layers(model)
    mods = dict(model.named_modules())
    return {n: mods[n].weight.numel() for n in layers}


def clone_fp32_model(fp_model):
    """FP 모델 깊은 복사 (디스크 재로드 대신 메모리서 fresh 모델, Codex)."""
    return copy.deepcopy(fp_model)


def make_ptq_model(fp_model, n_bits, device=DEVICE):
    """FP 모델 → fresh PTQ 모델 (deepcopy 후 ptq). 다회 run 효율."""
    return ptq(copy.deepcopy(fp_model), n_bits, device=device)


def fisher_diag(model, calib_loader, layers, n_batches=4, device=DEVICE):
    """층별 proxy = 배치평균 그래디언트의 제곱노름 (1/nb)·Σ_b ‖∇_{W_l}(mean-CE)‖² (= mean-CE grad-norm²).
    ⚠ true Fisher E_x‖∇log p_θ‖²도, 정식 empirical-Fisher 대각합 (1/B)Σ_j‖∇ℓ_j‖²도 아님 — true-label 사용 +
    제곱 전 배치평균이라 batch-size 의존(batch=128 고정이라 층간 상대랭킹엔 무해). proxy 비교용. PTQ 위, backward 1회/배치."""
    crit = nn.CrossEntropyLoss()
    mods = dict(model.named_modules())
    prev = {n: mods[n].weight.requires_grad for n in layers}   # 복원용 (Codex #4)
    for n in layers:
        mods[n].weight.requires_grad_(True)
    model.eval()
    acc = {n: 0.0 for n in layers}; nb = 0
    for i, (x, y) in enumerate(calib_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        crit(model(x), y).backward()
        for n in layers:
            g = mods[n].weight.grad
            if g is not None:
                acc[n] += (g.detach() ** 2).sum().item()
        nb += 1
    for n in layers:
        mods[n].weight.requires_grad_(prev[n])   # requires_grad 원복 (Codex #4)
    return {n: acc[n] / max(nb, 1) for n in layers}


def isolated_output_delta(fp_model, n_bits, calib_loader, layers, n_batches=4, device=DEVICE):
    """그 층 *하나만* 양자화했을 때 최종 logit 변화 ‖Δlogit‖² (forward-only, FP 위). RQ6 proxy."""
    mods = dict(fp_model.named_modules())
    scales = compute_scales(fp_model, n_bits)
    qmax = 2 ** (n_bits - 1) - 1
    fp_model.eval()
    xs, fp_logits = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(calib_loader):
            if i >= n_batches:
                break
            x = x.to(device); xs.append(x); fp_logits.append(fp_model(x).detach())
    out = {}
    for n in layers:
        w = mods[n].weight; orig = w.data.clone(); s = scales[n].to(w.device)
        try:                                          # 에러나도 fp_model 복원 보장 (Codex)
            with torch.no_grad():
                w.data = torch.clamp(torch.round(orig / s), -qmax, qmax) * s
                out[n] = sum((fp_model(x) - fl).pow(2).sum().item() for x, fl in zip(xs, fp_logits))
        finally:
            w.data = orig
    return out


def proxy_scores(ptq_model, fp_model, n_bits, calib_loader, layers=None, n_batches=4, device=DEVICE):
    """층별 5종 proxy: δᵀHδ·‖Hδ‖²(+부호)·Fisher·weight-error·isolated-output. 전부 학습 0(PTQ 위)."""
    layers = layers or list_quant_layers(ptq_model)
    deltas = quant_error(ptq_model)
    werr = {n: deltas[n].pow(2).sum().item() for n in layers}
    fisher = fisher_diag(ptq_model, calib_loader, layers, n_batches, device)
    iso = isolated_output_delta(fp_model, n_bits, calib_loader, layers, n_batches, device)
    out = {}
    for n in layers:
        hv = hvp_proxies(ptq_model, n, deltas[n], calib_loader, n_batches, device)
        out[n] = dict(dtHd=hv['dtHd'], normHd2=hv['normHd2'], sign=hv['sign'],
                      fisher=fisher[n], werr=werr[n], iso_out=iso[n])
    return out


def spearman(a, b):
    """Spearman 순위상관 ρ (scipy). 상수 벡터면 정의 안 됨 → np.nan (0으로 위장 안 함, Codex #5)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    rho, _ = spearmanr(a, b)
    return float(rho)


def kendall(a, b):
    """Kendall τ (§7 — Spearman 보강). 상수 벡터면 np.nan."""
    from scipy.stats import kendalltau
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    return float(kendalltau(a, b)[0])


def inversion_strength(short_recov, plateau_recov):
    """역전 강도 = 1 − Spearman(단기 회복 순위, 수렴 회복 순위). 같은 층 순서 리스트."""
    return 1.0 - spearman(short_recov, plateau_recov)


# --- S1.2 정밀판: 곡률 스케일 점검 + 유의성(순열·부트스트랩) ---
def _mean_hvp(model, layer_name, v, calib_loader, n_batches=4, device=DEVICE):
    """calib 여러 배치 평균 H_l·v (power iteration용)."""
    crit = nn.CrossEntropyLoss()
    s, n = None, 0
    for i, (x, y) in enumerate(calib_loader):
        if i >= n_batches:
            break
        Hv = hvp_layer(model, layer_name, v, x, y, crit, device)
        s = Hv if s is None else s + Hv; n += 1
    return s / max(n, 1)


def lambda_max_layer(model, layer_name, calib_loader, n_iter=15, n_batches=4, device=DEVICE):
    """층 Hessian 최대-크기 고유값 추정(power iteration via HVP).
    η·|λ_max|<1 이어야 §4.2 닫힌형태 (1-ηλ)^t 가 단조(발산/진동 아님) → lr 타당성 1회 점검."""
    W = dict(model.named_modules())[layer_name].weight
    v = torch.randn_like(W); v = v / (v.norm() + 1e-12)
    lam = 0.0
    for _ in range(n_iter):
        Hv = _mean_hvp(model, layer_name, v, calib_loader, n_batches, device)
        lam = float((v * Hv).sum().item())          # Rayleigh quotient (v 단위벡터)
        nv = float(Hv.norm().item())
        if nv < 1e-20:
            break
        v = Hv / nv
    return lam


def perm_pvalue_related(short_recov, plateau_recov, n_perm=2000, seed=0):
    """순열검정(보조 지표): 귀무(층 라벨 무작위)에서 단기·수렴 순위가 *우연히* 이만큼 양의 상관일 확률.
    p 작음 = 단기·수렴 순위가 양의 관련(=랜덤 아님)일 뿐 — *역전 유의성은 아님*. real_inversion 판정엔 안 씀
    (그건 rank_stability + bootstrap CI + breakdown gate가 함). short-long rank relation 보조용. 21층 셔플."""
    rng = np.random.default_rng(seed)
    a = np.asarray(short_recov, float); b = np.asarray(plateau_recov, float)
    obs = spearman(a, b)
    if np.isnan(obs):
        return dict(rho=float('nan'), p_value=float('nan'))
    cnt = 0
    for _ in range(n_perm):
        r = spearman(a, rng.permutation(b))
        if not np.isnan(r) and r >= obs:
            cnt += 1
    return dict(rho=float(obs), p_value=float((cnt + 1) / (n_perm + 1)))


def noise_floor_matched(per_seed, n_split=400, seed=0):
    """seed-*평균* 순위의 재현성 기반 noise floor = 1 − E[spearman(half_a_mean, half_b_mean)].
    split-half(무작위로 seed를 둘로 갈라 각 절반 평균순위 간 상관). inversion은 seed-평균 벡터로 계산되므로
    단일seed pairwise rank_stability(노이즈 full → noise 과대평가 → 게이트 과보수)보다 *같은 averaging level*에 정렬됨.
    (감사 지적: ci_lo[seed-평균]과 noise_inv[단일seed]의 평균화 불일치 교정.) n<2면 nan."""
    rng = np.random.default_rng(seed)
    S = np.asarray(per_seed, float); n = S.shape[0]
    if n < 2:
        return float('nan')
    h = n // 2; rs = []
    for _ in range(n_split):
        idx = rng.permutation(n)
        r = spearman(S[idx[:h]].mean(0).tolist(), S[idx[h:]].mean(0).tolist())
        if not np.isnan(r):
            rs.append(r)
    return (1.0 - float(np.mean(rs))) if rs else float('nan')


def bootstrap_inversion(short_per_seed, plateau_per_seed, n_boot=2000, seed=0):
    """seed 부트스트랩으로 inversion_strength 점추정+95%CI (자의적 SNR 컷 대신).
    입력 = [seed][layer] 2D 리스트. real_inversion 판정 = rank_stability + 이 부트 CI(ci_lo>noise_inv)
    + breakdown gate (perm_pvalue_related은 보조). seed 적으면 CI 넓음 → 5+ 권장."""
    rng = np.random.default_rng(seed)
    S = np.asarray(short_per_seed, float); P = np.asarray(plateau_per_seed, float)
    n_seed = S.shape[0]
    point = inversion_strength(S.mean(0).tolist(), P.mean(0).tolist())
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seed, n_seed)
        iv = inversion_strength(S[idx].mean(0).tolist(), P[idx].mean(0).tolist())
        if not np.isnan(iv):
            boots.append(iv)
    if not boots:
        return dict(inversion=float(point), ci_lo=float('nan'), ci_hi=float('nan'), n_boot=0)
    return dict(inversion=float(point), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), n_boot=len(boots))


def partial_spearman(x, y, z):
    """z(층 크기 N 등)를 통제한 x–y 부분 순위상관 = (r_xy − r_xz·r_yz)/sqrt((1−r_xz²)(1−r_yz²)).
    proxy↔회복 상관이 *곡률* 때문인지 *큰 층(N)* 때문인지 가른다(size 교란 제거, claude.ai②)."""
    rxy = spearman(x, y); rxz = spearman(x, z); ryz = spearman(y, z)
    if any(np.isnan(v) for v in (rxy, rxz, ryz)):
        return float('nan')
    denom = float(np.sqrt(max((1 - rxz**2) * (1 - ryz**2), 0.0)))
    return float((rxy - rxz * ryz) / denom) if denom > 1e-12 else float('nan')


def hvp_finite_diff(model, layer_name, delta, x, y, eps=1e-2, criterion=None, device=DEVICE):
    """유한차분 Hδ ≈ [∇L(W+εδ) − ∇L(W−εδ)]/(2ε). 반환 = Hδ 텐서.
    ⚠ fake-quant(QConv2d) 모델에 쓰면 함정: forward가 ±εδ를 *재양자화*(round)해 FD가 dense H·δ가 아니라
      sparse 재양자화 점프를 잰다 → hvp_layer와 cos≪1 거짓경보(claude.ai). 정식 실모델 교차검증은
      plain(비양자화) 모델 weight=wq0 둘레에서 float64로 ±εδ. hvp_layer 자체는 toy(float64 rel 3.8e-7)
      + S0(실모델 finite·non-zero·PSD부호)로 이미 검증됨 — 이 함수는 그 정식 FD(11월)용 보관."""
    criterion = criterion or nn.CrossEntropyLoss()
    W = dict(model.named_modules())[layer_name].weight
    x, y, delta = x.to(device), y.to(device), delta.to(device)
    model.eval()
    def grad_at(sgn):
        orig = W.data.clone(); prev = W.requires_grad
        W.data = orig + sgn * eps * delta; W.requires_grad_(True)
        try:
            g = torch.autograd.grad(criterion(model(x), y), W)[0].detach().clone()
        finally:
            W.data = orig; W.requires_grad_(prev)
        return g
    return ((grad_at(1.0) - grad_at(-1.0)) / (2 * eps)).detach()


def select_subset(scores, costs, budget_ratio, by='normHd2'):
    """proxy-top-k: by 점수 내림차순으로 예산(budget_ratio×총비용)까지 greedy (S2)."""
    budget = budget_ratio * sum(costs.values())
    key = lambda n: (scores[n][by] if isinstance(scores[n], dict) else scores[n])
    chosen, c = [], 0.0
    for n in sorted(scores, key=key, reverse=True):
        if c + costs[n] <= budget:
            chosen.append(n); c += costs[n]
    return chosen
