# report_assets — 색인

보고서 본문을 쓸 때 쓰는 재료 묶음이다. 모든 수치는
`python analysis/run_all.py` 한 번으로 다시 만들어진다(GPU 불필요).

```
report_assets/
├── FINDINGS.md     핵심 문서. 질문 단위로 묶은 발견
├── numbers.json    본문에 인용될 모든 수치 (키-값)
├── figures/        PNG 300dpi + 같은 이름의 .csv (그림의 원자료)
└── tables/         각 표의 .csv와 .md
```

## 읽는 규칙

- **[진단용]** 태그가 붙은 것은 테스트 사람의 라벨이나 메타데이터
  (`Origin`, `WhichEye`, day)를 쓴 분석이다. 메인 LOPO 수치와 섞지 않는다.
  파일명에 `diag_`가 붙어 있다.
- 모든 비교는 15 fold 대응표본 t와 p값을 함께 적었다. p ≥ 0.05인 차이는
  "개선"이라고 쓰지 않았다.
- 실행 이름: `runB` = `results/final` (batch 128, dropout 0.5, 20 epoch).
  `runA` = `results/base`. **runA의 `predictions.npz`는 제 `lopo.json`과
  11/15 fold에서 어긋나므로 예측 기반 분석에 쓰지 않았다**(Phase 0에서 발견).

## 그림

| 파일 | 답하는 질문 | 원자료 | 생성 |
|---|---|---|---|
| `12_person_offset.png` | 사람별 치우침이 얼마나 크고 fold 난이도를 예측하나 | `12_person_offset.csv` | `analysis/a1_offset.py` |
| `diag_13_calibration.png` | 테스트 라벨 k장으로 offset을 얼마나 지우나 | `diag_13_calibration.csv` | `analysis/a2_calibration.py` |
| `14_regression_dilution.png` | '수축'은 모델의 성향인가 회귀의 성질인가 | `14_regression_dilution.csv` | `analysis/a3_dilution.py` |
| `15_throughput.png` | 배치와 패킹 중 무엇이 처리량 레버인가 | `15_throughput.csv` | `analysis/a9_efficiency.py` |
| `16_epoch_selection.png` | val로 epoch을 고르는 것이 이득인가 | `16_epoch_selection.csv` | `analysis/a4_selection.py` |
| `diag_17_eye_consistency.png` | 오차는 이미지 수준인가 장면 수준인가 | `diag_17_eye_consistency.csv` | `analysis/a5_eyes.py` |
| `diag_18_conditions.png` | day·head pose·gaze 방향에 따라 오차가 어떻게 다른가 | `diag_18_conditions.csv` | `analysis/a6_conditions.py` |
| `19_fold_difficulty.png` | fold 난이도는 실행마다 바뀌나 사람의 속성인가 | `19_fold_difficulty.csv` | `analysis/a7_difficulty.py` |
| `20_ensemble.png` | epoch·seed ensemble이 얼마나 줄이나 | `20_ensemble.csv` | `analysis/e2_ensemble.py` |
| `21_ablation.png` | 각 설계 결정이 실제로 기여하나 | `21_ablation.csv` | `analysis/e3_ablation.py` |
| `diag_22_labelfree.png` | 라벨 없는 신호가 offset을 알려 주나 | `diag_22_labelfree.csv` | `analysis/a10_labelfree.py` |
| `diag_23_zeroshot.png` | 라벨 0장으로 offset을 지울 수 있나 | `diag_23_zeroshot.csv` | `analysis/a11_zeroshot.py` |
| `24_interpretation.png` | 모델은 이미지의 어디를 보나 | `24_interpretation.csv` | `analysis/a8_interpret.py` |
| `diag_25_offset_nature.png` | offset은 해부학적 상수인가 분포 이동인가 | `diag_25_offset_nature.csv` | `analysis/a12_offset_nature.py` |
| `diag_26_variance_layers.png` | 남은 오차 중 얼마가 원리적으로 줄어드나 | `diag_26_variance_layers.csv` | `analysis/a13_noisefloor.py` |

## 표

| 파일 | 내용 | 생성 |
|---|---|---|
| `final_results` | 최종 결과 표 (baseline·이전 실행·최종 모델 단일/ensemble) | `analysis/final_table.py` |
| `e2_ensemble` | (a) 단일 (b) epoch-ens (c) seed-ens (d) 둘 다 | `analysis/e2_ensemble.py` |
| `e3_ablation` | 항목 · EDA의 예측 · Δ±SE · p · 일치 여부 · 흡수 여부 | `analysis/e3_ablation.py` |
| `a1_person_offset_runB` | 사람별 offset과 fold 오차 | `analysis/a1_offset.py` |
| `a2_calibration_runB` | few-shot calibration 곡선 [진단용] | `analysis/a2_calibration.py` |
| `a3_slopes_runB`, `a3_gaze_size_runB` | 회귀 기울기와 gaze 크기별 오차 | `analysis/a3_dilution.py` |
| `a4_epoch_selection_runB` | fold별 val 선택 / 마지막 / oracle epoch | `analysis/a4_selection.py` |
| `a5_eye_consistency_runB` | 좌우 눈 상관 [진단용] | `analysis/a5_eyes.py` |
| `a6_day_drift_runB`, `a6_headpose_bins_runB` | 조건별 오차 [진단용] | `analysis/a6_conditions.py` |
| `a7_run_rank_correlation`, `a7_offset_vs_folderr` | 난이도 재현성 | `analysis/a7_difficulty.py` |
| `a8_head_ablation`, `a8_person_probe` | head pose 망가뜨리기, person probe | `analysis/a8_interpret.py` |
| `a9_throughput_grid` | batch × per-gpu 처리량 | `analysis/a9_efficiency.py` |
| `a10_labelfree_signals_runB` | 라벨 없는 신호들의 상관 | `analysis/a10_labelfree.py` |
| `a11_zeroshot_runB`, `a11_perfold_runB` | 라벨 0장 재중심화 [진단용] | `analysis/a11_zeroshot.py` |
| `a12_h1_day_stability_runB` 외 2 | offset의 정체 검증 [진단용] | `analysis/a12_offset_nature.py` |
| `a13_variance_layers_runB` | 잔차 분산 3층 분해 [진단용] | `analysis/a13_noisefloor.py` |

## 실험 기록

- `PREREG.md` — GPU 실험 전에 적은 비교 대상·사전 예상·선택 기준 (commit `279b8b5`)
- `QUESTIONS.md` — 관찰에서 나온 질문 장부 (관찰 → 질문 → 가설 → 최소 실험 → 상태)
- `PLAN.md` — Phase 마감 (KST 절대 시각)
- `results/JOBS.md` — SLURM job ID·설정·소요·성공/실패와 원인
