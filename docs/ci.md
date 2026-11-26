# CI regression gate

1. Run your evaluation once on `main` and commit the results as a baseline:

    ```bash
    ragcheck run eval.yaml -o eval/baseline.json
    ```

2. In CI, run the same config and compare:

    ```yaml
    - run: ragcheck run eval.yaml --no-progress -o results/current.json
    - run: >
        ragcheck compare eval/baseline.json results/current.json
        --fail-on "recall@5<-0.02" --fail-on "faithfulness<-5%" --fail-on "latency_ms_mean>100"
    - run: ragcheck report results/current.json --html results/report.html --baseline eval/baseline.json
      if: always()
    - uses: actions/upload-artifact@v4
      if: always()
      with: {name: rag-report, path: results/}
    ```

## Rule syntax

`METRIC OP VALUE[%]`

- Left-hand side is the delta `current - baseline`; with `%` it is the relative delta
  `100 * (current - baseline) / |baseline|` (skipped when the baseline is 0).
- `OP` is one of `<`, `<=`, `>`, `>=`. The rule *fails* when the comparison is true.
- `METRIC` is any metric name, any run statistic (`latency_ms_mean`, `latency_ms_p95`,
  `errors`, ...), or `*` for every metric present in both runs.
- A named metric missing from either run is reported as `MISSING` and fails the gate.

Exit codes: `0` no violations, `1` at least one violation, `2` usage error (bad rule,
unreadable file). `--json` prints a machine-readable summary.

!!! tip "Keep the judge fixed"
    LLM-judge scores depend on the judge model and prompt. Use the same judge for baseline
    and current runs, and enable `cache:` so unchanged cases cost nothing to re-score.
