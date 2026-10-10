GROUND RULES FOR THIS REPO (non-negotiable)
1. Honesty: never invent, round up, or "estimate" any metric. Every number in README/RESULTS/model cards
   must be produced by code in this repo and written to results/*.json by a script. If a run was not done,
   write "not measured". Report confidence intervals (bootstrap, 1,000 resamples, seed fixed) for headline metrics.
2. Data: only public or synthetic data. No real personal data. Label synthetic data as synthetic everywhere.
   Public datasets: Lending Club (Kaggle), freMTPL2 (OpenML 41214/41215), UCI Statlog German Credit,
   OFAC SDN list (treasury.gov), OpenSanctions (CC BY-NC 4.0, attribute it). Do not commit raw datasets; download via scripts.
3. Reproducible: Python 3.11, uv or pip-tools lockfile, `make` targets for every step, fixed seeds, pytest for
   every module, GitHub Actions CI running tests + lint (ruff) on every PR.
4. No secrets in git: use .env (gitignored) + .env.example. Cost cap: stop and report if a run would exceed $10.
5. Plain code a junior engineer can explain: small modules, docstrings, type hints, no clever one-liners.
   After each phase write docs/walkthrough_phaseN.md explaining every file, design choice, and 8-10 likely
   interview questions with answers, so Rahul can explain the work in his own words.
6. Never claim a cloud service, model, or tool in docs unless the code actually ran against it in this repo.
7. Commit in small logical commits with clear messages. Open a PR per phase with a summary + results table.
