# 26_Capstone — Budget-Dependent Quantization Recovery

광운대 캡스톤 (24조 Sae-Bits). **양자화로 망가진 비전 모델을 *아주 짧게만* 재학습할 수 있을 때, 모델의 *어느 부분*을 학습해야 가장 많이 회복되는가 — 그 답이 *끝까지 학습한 답과 다른지(역전)*, 비트가 낮아질수록 그 차이가 *커지는지*를 학습 동역학으로 예측하고 측정으로 검증한다.**

> 연구·실험 설계 문서(03 연구계획서 · 04 실험계획서 · 05 sprint · 06 빌드맵)는 Obsidian vault가 진실원.
> **이 repo는 그 실험을 *실행*하는 코드.**

## 구조
```
26_Capstone/
├─ qat_engine.py     # 단일 진실원(엔진): 모델·데이터·양자화·partial QAT·HVP·proxy·노이즈바닥·로깅
├─ notebooks/        # 얇은 실행 노트북(페이즈당 1개) — 로직 X, sweep·플롯만
│   ├─ S0_tooling_noisefloor.ipynb     # 도구·바닥
│   ├─ S1_diagnostic_inversion.ipynb   # 척추: 역전 + proxy 예측
│   └─ (S2 전략 · S3 비트 · S4 GSB 다리 …)
├─ checkpoints/ data/ outputs/         # gitignore — 무거운 산출물은 Drive
├─ requirements.txt
└─ .gitignore
```

## 원칙
- **로직은 전부 `qat_engine.py` 한 곳.** 노트북은 얇게(import해서 호출만). 버그/수정은 엔진에서.
- 단일 경로 commit · fallback 미리 안 깖 · 메서드 점검(노이즈바닥·W8 sanity·W2 게이트)은 *절차*라 유지.

## Colab 워크플로우
```python
!git clone https://github.com/u-nsiq/26_Capstone.git
%cd 26_Capstone
!pip install -q -r requirements.txt
import torch; print("torch", torch.__version__)          # 버전 기록
from qat_engine import *
from google.colab import drive; drive.mount('/content/drive')   # baseline·결과 영구저장
ART = '/content/drive/MyDrive/26_Capstone'                 # data/ckpt/outputs 를 Drive로
```
- 코드 갱신: 내가 push → Colab에서 `!git pull` 한 줄.
- baseline(~76% FP32 CIFAR-100)은 **한 번 학습→Drive 캐시**, 이후 즉시 로드(재학습 0).

## S0 성공 기준 (안 나오면 다음 페이즈 무의미)
1. `roundtrip_test(8)` 통과 + **W8 PTQ ≈ 무손실** (manual conv 양자화 정확).
2. **W4 PTQ 갭 > 노이즈 바닥** (회복할 게 있다).
3. `ptq()` silent-skip assert 통과 (모든 대상 층이 *실제로* 양자화됨).
4. HVP 셀 finite·동작 + `δᵀHδ` 부호 로깅.

## 엔진에 박힌 결정 (요약)
manual additive-STE · per-channel · **고정 scale** / **momentum=0**(진단 run, vanilla GD) / PTQ **silent-skip 가드** / W8 sanity = **무손실 + round-trip**(torchao는 fc만 선택) / 노이즈바닥 = **단일층 recovery std** / BN stats 고정 / swap 후 device 이동. 근거는 `qat_engine.py` 상단 도크 + 05/06 문서.

## 비트 축 극단점(W1) 자산
GSB(1-bit ViT) 재현물은 별도 위치(`Research/Capstone/ref_code/GSB-Vision-Transformer`, `notebooks/01_GSB`). S4(밤3) 다리에서 연결.
