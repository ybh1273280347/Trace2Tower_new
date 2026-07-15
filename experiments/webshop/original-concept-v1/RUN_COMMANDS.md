# WebShop Run Commands

```powershell
Set-Location D:\Trace2Tower
$env:PYTHONPATH = ".;src"
$py = ".\.venv\Scripts\python.exe"
$train = "experiments/webshop/original-concept-v1/manifests/refinement-train.jsonl"
$validation = "experiments/webshop/event-tower-v2/manifests/validation.jsonl"
$test = "experiments/webshop/event-tower-v2/manifests/test.jsonl"
$v0 = "artifacts/trace2tower/original-concept-v1/p100/full/tower.json"
$t1 = "artifacts/trace2tower/original-concept-v1/p100/pareto-v1/tower.json"
```

## Train Feedback

```powershell
& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split train --method no_skill --manifest webshop=$train --repeat-id 0 --repeat-id 1 --repeat-id 2 --run-id webshop-original-concept-v1-refinement-train-flash-noskill-r3 --agent-model deepseek-v4-flash --episode-concurrency 3 --api-concurrency 3 2>&1 | Tee-Object artifacts/logs/refinement-train-noskill-r3.log

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split train --method trace2tower --artifact webshop=$v0 --manifest webshop=$train --repeat-id 0 --repeat-id 1 --repeat-id 2 --run-id webshop-original-concept-v1-refinement-train-flash-tower-v0-r3 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 3 --api-concurrency 3 2>&1 | Tee-Object artifacts/logs/refinement-train-tower-v0-r3.log
```

## Offline Pareto Build

```powershell
& $py scripts/experiments/analyze/build_refinement_pareto_selection.py --tower $v0 --baseline-run artifacts/runs/webshop-original-concept-v1-refinement-train-flash-noskill-r3 --tower-run artifacts/runs/webshop-original-concept-v1-refinement-train-flash-tower-v0-r3 --output experiments/webshop/original-concept-v1/refinement/usage-pareto.json

& $py scripts/experiments/analyze/build_structural_pareto.py --tower $v0 --candidate-clusters artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/clusters.json --spectral artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/spectral.npz --transition artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/transition.npz --output experiments/webshop/original-concept-v1/refinement/structural-pareto.json

& $py scripts/experiments/build/build_pareto_refined_skills.py --tower $v0 --preprocessed artifacts/trace2tower/original-concept-v1/p100/refinement-v1/preprocessed.jsonl --candidate-clusters artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/clusters.json --structural-pareto experiments/webshop/original-concept-v1/refinement/structural-pareto.json --output-dir artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills

& $py scripts/experiments/build/build_trace2tower_index.py --cards artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/rendered-cards.json --config configs/experiments/webshop_trace2tower.yaml --output artifacts/trace2tower/original-concept-v1/p100/pareto-v1/index.json --direct-mid-top-k 8

& $py scripts/experiments/build/build_tower_snapshot.py --benchmark webshop --version v1 --input artifacts/trace2tower/original-concept-v1/p100/refinement-v1/preprocessed.jsonl --clusters artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/mid-clusters.json --high-paths artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/high-paths.json --cards artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/rendered-cards.json --index artifacts/trace2tower/original-concept-v1/p100/pareto-v1/index.json --config configs/experiments/webshop_trace2tower.yaml --output $t1

& $py scripts/experiments/analyze/build_refinement_action_plan.py --tower $v0 --refined-tower $t1 --candidate-clusters artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/clusters.json --usage-pareto experiments/webshop/original-concept-v1/refinement/usage-pareto.json --structural-pareto experiments/webshop/original-concept-v1/refinement/structural-pareto.json --refined-build-report artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/report.json --output experiments/webshop/original-concept-v1/refinement/action-plan.json
```

## Method Configs

```text
NoSkill:   configs/experiments/webshop_no_skill.yaml
Manual:    configs/experiments/webshop_manual_skill.yaml
Global:    configs/experiments/webshop_global_e2e.yaml
SkillX:    configs/experiments/webshop_skillx.yaml
Tower V0:  configs/experiments/webshop_trace2tower_runtime.yaml
Tower T1:  configs/experiments/webshop_trace2tower_refinement_v1_runtime.yaml
```

## Validation

```powershell
& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split dev --method no_skill --manifest webshop=$validation --repeat-id 0 --run-id webshop-original-concept-v1-validation-flash-noskill-r1 --agent-model deepseek-v4-flash --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split dev --method trace2tower --artifact webshop=$v0 --manifest webshop=$validation --repeat-id 0 --run-id webshop-original-concept-v1-validation-flash-v0-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split dev --method trace2tower --artifact webshop=$t1 --manifest webshop=$validation --repeat-id 0 --run-id webshop-original-concept-v1-validation-flash-pareto-v1-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_refinement_v1_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 3 --api-concurrency 3
```

## Test-A Baselines

```powershell
& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method no_skill --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-noskill-r1 --agent-model deepseek-v4-flash --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method manual_skill --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-manual-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_manual_skill.yaml --fixed-skill-id manual_webshop_constraint_v1 --fixed-skill-context-file experiments/webshop/event-tower-v2/manual-skill.md --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method global_e2e_gpt --artifact webshop=artifacts/global_e2e/event-tower-v2/p50/library.json --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-global-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_global_e2e.yaml --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method skillx --artifact webshop=artifacts/skillx/event-tower-v2/p50/execution/library.json --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-skillx-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_skillx.yaml --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$v0 --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-v0-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 3 --api-concurrency 3

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$t1 --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-pareto-v1-r2 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_refinement_v1_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 1 --api-concurrency 1 2>&1 | Tee-Object artifacts/logs/test-a-pareto-v1-r2.log
```

## Graph-aware Retrieval Diagnosis

```powershell
& $py -m scripts.experiments.build.build_tower_graph_profile --tower $t1 --preprocessed artifacts/trace2tower/original-concept-v1/p100/refinement-v1/preprocessed.jsonl --output artifacts/trace2tower/original-concept-v1/p100/pareto-v1/graph-retrieval-profile.json

& $py scripts/experiments/analyze/diagnose_tower_retrieval.py

& $py -m scripts.experiments.build.build_pareto_refined_skills --tower $v0 --preprocessed artifacts/trace2tower/original-concept-v1/p100/refinement-v1/preprocessed.jsonl --candidate-clusters artifacts/trace2tower/original-concept-v1/p100/refinement-v1/graph/clusters.json --structural-pareto experiments/webshop/original-concept-v1/refinement/structural-pareto.json --output-dir artifacts/trace2tower/original-concept-v1/p100/pareto-high6/skills --max-high-path-length 6

Get-FileHash artifacts/trace2tower/original-concept-v1/p100/pareto-v1/skills/high-paths.json, artifacts/trace2tower/original-concept-v1/p100/pareto-high6/skills/high-paths.json -Algorithm SHA256
```

The max4 and max6 High path files are identical under the frozen P100 support
contract, so there is no High-length rollout command until the structural
comparison produces different skills.

## Graph-aware Test-A

```powershell
& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$t1 --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-graph-cap3-r2 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_final_runtime.yaml --direct-mid-top-k 3 --episode-concurrency 3 --api-concurrency 3 2>&1 | Tee-Object artifacts/logs/test-a-graph-cap3-r2.log

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$t1 --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-graph-cap8-r2 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_final_runtime.yaml --direct-mid-top-k 8 --episode-concurrency 3 --api-concurrency 3 2>&1 | Tee-Object artifacts/logs/test-a-graph-cap8-r2.log
```

The CLI option retains its historical name, but with `retrieval_strategy:
graph` it sets `mid_context_budget`. This is a total injected Mid budget,
including the active High path node and its directed successor.

```powershell
& $py -m scripts.experiments.analyze.analyze_graph_retrieval_test --noskill-run artifacts/runs/webshop-original-concept-v1-test-flash-noskill-r1 --v0-run artifacts/runs/webshop-original-concept-v1-test-flash-p100-full-cap8-r1 --legacy-cap8-run artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-cap8-r1 --legacy-cap3-run artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-cap3-r1 --graph-cap3-run artifacts/runs/webshop-original-concept-v1-test-a-flash-graph-cap3-r2 --graph-cap8-run artifacts/runs/webshop-original-concept-v1-test-a-flash-graph-cap8-r2 --output experiments/webshop/original-concept-v1/refinement/graph-retrieval-test-a.json
```

## Final Algorithm Follow-ups

```powershell
& $py -m scripts.experiments.build.build_tower_graph_profile --tower $v0 --preprocessed artifacts/trace2tower/original-concept-v1/p100/mixed/preprocessed.jsonl --output artifacts/trace2tower/original-concept-v1/p100/full/graph-retrieval-profile.json

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$v0 --manifest webshop=$test --repeat-id 0 --run-id webshop-original-concept-v1-test-a-flash-v0-graph-cap3-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_v0_graph_runtime.yaml --direct-mid-top-k 3 --episode-concurrency 2 --api-concurrency 2

& $py scripts/experiments/run/run_matrix.py --benchmark webshop --split test --method trace2tower --artifact webshop=$t1 --manifest webshop=experiments/webshop/original-concept-v1/manifests/test-b.jsonl --repeat-id 0 --run-id webshop-original-concept-v1-test-b-flash-final-graph-cap3-r1 --agent-model deepseek-v4-flash --method-config configs/experiments/webshop_trace2tower_final_runtime.yaml --direct-mid-top-k 3 --episode-concurrency 2 --api-concurrency 2
```

## Statistics

```powershell
$runRoots = @(
  "artifacts/runs/webshop-original-concept-v1-test-a-flash-noskill-r1",
  "artifacts/runs/webshop-original-concept-v1-test-a-flash-v0-r1",
  "artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-r2"
)
foreach ($root in $runRoots) {
  $rows = @(Get-ChildItem $root -Recurse -Filter results.jsonl -ErrorAction SilentlyContinue | ForEach-Object { Get-Content $_.FullName } | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
  [pscustomobject]@{
    Run = $root
    Rows = $rows.Count
    UniqueKeys = ($rows | Group-Object sample_id,repeat_id).Count
    MeanReward = [math]::Round(($rows | Measure-Object primary_score -Average).Average, 6)
    FullSuccess = ($rows | Where-Object { [double]$_.primary_score -ge 0.999 }).Count
    MeanSteps = [math]::Round(($rows | Measure-Object steps -Average).Average, 4)
    MeanInvalid = [math]::Round(($rows | Measure-Object invalid_actions -Average).Average, 4)
    MeanInputTokens = [math]::Round(($rows | Measure-Object input_tokens -Average).Average, 2)
  }
}
```

```powershell
& $py scripts/experiments/analyze/analyze_refinement_test.py --structural-run artifacts/runs/webshop-original-concept-v1-test-a-flash-pareto-v1-r2 --noskill-run artifacts/runs/webshop-original-concept-v1-test-a-flash-noskill-r1 --previous-tower-run artifacts/runs/webshop-original-concept-v1-test-a-flash-v0-r1 --output experiments/webshop/original-concept-v1/refinement/test-a-pareto-v1.json *> artifacts/logs/test-a-pareto-v1-statistics.log
```

## Live Monitor

```powershell
$runId = "webshop-original-concept-v1-test-a-flash-pareto-v1-cap8-r1"
$root = "artifacts/runs/$runId"
if (-not (Test-Path $root)) { throw "Run directory does not exist: $root" }
while ($true) {
  $rows = @(Get-ChildItem $root -Recurse -Filter results.jsonl -ErrorAction SilentlyContinue | ForEach-Object { Get-Content $_.FullName } | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json })
  $errors = @(Get-ChildItem $root -Recurse -Filter errors.jsonl -ErrorAction SilentlyContinue | ForEach-Object { Get-Content $_.FullName } | Where-Object { $_ })
  Clear-Host
  [pscustomobject]@{ Time = Get-Date -Format HH:mm:ss; Rows = $rows.Count; UniqueKeys = ($rows | Group-Object sample_id,repeat_id).Count; MeanReward = if ($rows.Count) { [math]::Round(($rows | Measure-Object primary_score -Average).Average, 6) } else { 0 }; FullSuccess = ($rows | Where-Object { [double]$_.primary_score -ge 0.999 }).Count; Errors = $errors.Count } | Format-List
  Start-Sleep 5
}
```
