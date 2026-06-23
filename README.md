# Seismic Waveform Generation via VAE + HiFi-GAN

COSE362 K08 프로젝트  
Conditional VAE와 HiFi-GAN을 활용한 지진파 합성 및 EQTransformer 데이터 증강 실험

---

## 프로젝트 개요

STEAD 데이터셋의 지진 파형을 학습 데이터로 사용하여, Conditional VAE로 스펙트로그램을 생성하고 HiFi-GAN으로 파형을 복원하는 파이프라인을 구축하였다. 최종적으로 생성된 합성 M≥4 지진파를 EQTransformer 학습 데이터 증강에 활용하여 탐지 성능 향상 여부를 검증한다.

---

## 파이프라인

```
STEAD Dataset (chunk1, chunk3, chunk4)
         │
         ▼
01. 베이스라인 cVAE 학습 + 스펙트로그램 추출
         │
         ▼
02. HiFi-GAN 학습 (M≥4 데이터 파인튜닝)
         │
         ▼
03. HiFi-GAN 성능 평가 (vs Griffin-Lim)
         │
04. FiLM VAE 파인튜닝 (조건부 생성 강화)
         │
         ▼
05. 합성 지진파 대량 생성 (M=4,5,6,7)
         │
         ▼
06. EQTransformer 증강 실험 (r=0,1,10,50)
```

---

## 코드 구성

| 파일                                    | 설명                                                   |
| --------------------------------------- | ------------------------------------------------------ |
| `01_vae_spectrogram_extract.ipynb`      | CGM-GM 논문 기반 cVAE 학습 및 스펙트로그램 추출        |
| `02_hifigan_train_finetune_m4.ipynb`    | 지진파 도메인 맞춤 HiFi-GAN 학습 (M≥4 파인튜닝)        |
| `03_hifigan_eval_finetune_m4.ipynb`     | HiFi-GAN 성능 평가 (Griffin-Lim 비교)                  |
| `04_film_vae_train_filmfixed.ipynb`     | FiLM 조건화 VAE 학습 (MR-STFT + Ranking Loss)          |
| `05_generate_synthetic_waveforms.ipynb` | FiLM VAE + HiFi-GAN 파이프라인으로 합성 파형 대량 생성 |
| `06_eqtransformer_augmentation.ipynb`   | 합성 파형 기반 EQTransformer 증강 실험                 |

---

## 모델 상세

### 01. 베이스라인 cVAE (`01_vae_spectrogram_extract.ipynb`)

CGM-GM 논문의 cVAE 구조를 포팅하여 STEAD 데이터에 적용하였다.

- **입력:** log₁₀ linear STFT amplitude spectrogram (81 freq × 131 time)
- **조건 변수 (6개):** source magnitude, source latitude/longitude, receiver latitude/longitude, source depth (km)
- **학습 데이터:** M<4 지진 이벤트 (층화 샘플링 5,000개)
- **손실 함수:** Reconstruction Loss + KL Divergence
- **출력:** VAE 복원 스펙트로그램 → HiFi-GAN 학습용으로 저장

### 02–03. HiFi-GAN (`02`, `03`)

오디오용 HiFi-GAN을 지진파 도메인에 맞게 수정하였다.

| 항목                  | 원본 HiFi-GAN          | 수정 후                            |
| --------------------- | ---------------------- | ---------------------------------- |
| 입력                  | Mel-spectrogram (80ch) | log₁₀ linear STFT amplitude (81ch) |
| MPD periods           | [2, 3, 5, 7, 11]       | [10, 20, 50]                       |
| MSD sub-discriminator | 3개                    | 2개                                |
| 손실 함수             | Mel Loss               | STFT Loss (λ=45)                   |
| Gradient clipping     | 없음                   | max_norm=1.0                       |

- **학습:** AdamW, lr=0.0002, batch=4, 200 epochs, lr_decay=0.999
- **평가 지표:** STFT Loss, SNR (dB), Cross-Correlation, FAS Residual (5–15 Hz)
- **비교 기준선:** Griffin-Lim 알고리즘

### 04. FiLM VAE (`04_film_vae_train_filmfixed.ipynb`)

베이스라인 cVAE의 조건부 생성 능력을 강화하기 위해 FiLM(Feature-wise Linear Modulation)을 GRU 인코더에 적용하였다.

- **FiLM:** 조건 벡터에서 (γ, β)를 생성하여 각 GRU 레이어의 활성화를 변조
- **추가 손실 함수:**
  - Multi-Resolution STFT Loss (fft_size: 64, 128, 256)
  - Ranking Loss: M이 클수록 PGV가 커야 한다는 단조 제약
- **학습 데이터:** M<4 (20,000개), 층화 샘플링
- **검증:** PGV 단조성 확인, 잠재공간 PCA 시각화

### 05. 합성 파형 생성 (`05_generate_synthetic_waveforms.ipynb`)

FiLM VAE → HiFi-GAN 파이프라인으로 M=4, 5, 6, 7 조건의 합성 파형을 대량 생성한다.

- **생성 조건:** magnitude M, source/receiver latitude/longitude, source depth (km)
- **생성량:** 실제 M≥4 데이터 대비 최대 r=100배
- **검증:** M별 PGV 단조성 확인

### 06. EQTransformer 증강 실험 (`06_eqtransformer_augmentation.ipynb`)

합성 파형이 EQTransformer 탐지 성능 향상에 기여하는지 검증한다.

- **모델:** EQTransformer 경량 구현 (Conv1D Encoder → LSTM → Self-Attention → Binary Classifier)
- **실험 설계:**
  - TSTR (Train on Synthetic, Test on Real): 합성 데이터 품질 검증
  - Augmentation: 실제 데이터 + r배 합성 데이터 (r = 0, 1, 10, 50)
- **평가 지표:** Accuracy, Precision, Recall (M≥4 탐지 이진 분류)

---

## 데이터셋

**STEAD** (STanford EArthquake Dataset)

- 사용 파일: chunk1, chunk3, chunk4 (HDF5 + CSV)
- 분할: M<4 → VAE 학습/검증, M≥4 → HiFi-GAN 학습 및 테스트
- SR=100Hz, 60초 파형 (6,000 samples)

데이터는 Google Drive에 마운트하여 사용한다 (`/content/drive/MyDrive/ML_Project/`).

---

## 실행 환경

- Google Colab (GPU 필수)
- Python 3.10+
- 실험 기록: [wandb](https://wandb.ai) (`seismic-hifigan` 프로젝트)

```bash
pip install -r requirements.txt
```

Google Colab Secrets에 다음 키를 등록해야 한다:

- `GITHUB_TOKEN`
- `WANDB_API_KEY`
