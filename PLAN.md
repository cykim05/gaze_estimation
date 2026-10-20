# PLAN — 10시간 실험·분석 계획

**T0 = 2026-10-20 11:31 KST** (UTC 02:31)

| Phase | 마감 (KST) | 내용 |
|---|---|---|
| Phase 0 | **12:31** | 사전 발견 4개 재현, PREREG/QUESTIONS 작성, 로깅 패치, smoke job |
| Phase 1 | **13:31** | B0 처리량 · B1 배치 정확도 (GPU 2장 분할) |
| Phase 2 | **18:31** | 본 실험 큐 P1~P5 + A8. **이후 새 GPU job 제출 금지** |
| — | **19:01** | 돌고 있는 job 대기 한계. 이후 취소 |
| Phase 3 | **20:31** | 결과 통합 (E2 ensemble, E3 ablation, 최종 표) |
| Phase 4 | **21:31** | 패키징·검수·push. **report_assets/ 완성 마감** |

분석 트랙(CPU, A1~A9)은 Phase 0부터 GPU와 병렬로 상시 진행한다.

## 자원 규칙 (절대)

- GPU job은 `sbatch`만. login 노드는 npz를 읽는 짧은 CPU 분석만.
- **동시 RUNNING GPU job ≤ 2** (job당 `--gres=gpu:1`, array는 `%2`).
- 제출 전 `squeue -u $USER` 확인. `results/JOBS.md`에 누적 기록.
- cupy memory pool 상한으로 카드 메모리 70% 미만 사용. `--min-free`로 남의 카드 회피.
- 실패 job은 원인 고쳐 1회만 재제출, 재실패 시 잘라낸다.

## 예산 계산 (Phase 1 종료 후 갱신)

기존 로그 기준 15-fold 1회(20 epoch, batch 128) = 47~70 GPU·분 → GPU 2장에서 25~35분.
Phase 2 가용 시간 5시간 × 2장 = 600 GPU·분 → 15-fold 약 8~12회.

- P1 (seed 3개) = 3회
- P2 (ablation 4종) = 4회
- P3 (ablation 2종) = 2회
- P4 (seed 2개 추가) = 2회
- P5 (추가 seed) = 1~2회
- A8 (eval-only) = 0.3회

합계 약 12~13회. **배치 탐색에서 더 큰 배치가 선택되면 회당 시간이 줄어 여유가 생긴다.**
시간이 모자라면 P5 → P4 → P3 순으로 자른다.
