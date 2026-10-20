# JOBS — SLURM job 기록

규칙: 동시 RUNNING GPU job ≤ 2. job당 `--gres=gpu:1`. array는 `%2`.

| job | phase | 설정 | GPU | 제출(KST) | 소요 | 결과 |
|---|---|---|---|---|---|---|
| 1787 | P0 | smoke 2fold×1ep | 1 | 11:33 | 1m | 실패 — Samples.take에 cupy 인덱스 (numpy person과 충돌) |
| 1788 | P0 | smoke 재시도 | 1 | 11:36 | 1m | 실패 — SLURM이 남이 쓰던 카드 배정, --min-free 8 과함 |
| 1789 | P0 | smoke 재시도2 | 1 | 11:39 | 2m | **성공** — 로깅 배열 17종 + 특징 저장 확인, 정합성 OK |
| 1790 | P1 | B0 처리량 격자 | 1 | 11:44 | 0m | 실패 — SmiSampler._stop이 Thread._stop을 가림 |
| 1791 | P1 | B1 배치×seed 12회 | 1 | 11:44 | 진행 | 4 batch × 3 seed, 20 epoch, 검증만 |
| 1792 | P1 | B0 재제출 | 1 | 11:47 | 진행 | batch{128,256,512,1024} × per-gpu{1,2,3} |
| 1792 | P1 | B0 처리량 격자 | 1 | 11:47 | 3m | **성공** — batch는 레버 아님, 패킹은 b128에서만 1.49× |
| 1793 | P2 | P1 seed 0, 15-fold, b128 | 1 | 11:47 | 진행 | B1 확정 전 선제 제출 (b128이 선택될 가능성 높음) |
| 1791 | P1 | B1 배치×seed 12회 | 1 | 11:44 | 22m | **성공** — 네 배치 모두 구분 불가, b128 선택 |
