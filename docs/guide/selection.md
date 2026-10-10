# Participants, sessions and runs

`--participant-label` and `--session-id` select participants and sessions, as in XCP-D; for `xcpd` and `nirspipe`, `--task-id` selects the task. Each session is analysed on its own; without `--session-id`, every session in `INPUT` is.

## BIDS filter files

For files that differ in other BIDS entities, such as several runs or acquisitions, `--bids-filter-file` takes the same JSON file as XCP-D. Its `"bold"` entry (`"nirs"` for nirspipe) names the entities the files must carry; a list accepts any of its values and `null` requires the entity to be absent. Task, space or session given there replace `--task-id`, `--space` and `--session-id`.

```json
{"bold": {"acquisition": "multiband", "run": 1}}
```

## Several runs

A participant with several runs left after filtering has each run analysed on its own (`01_run-1`, `01_run-2`), with a warning in the report; group comparisons stop instead, since they need one matrix per participant. XCP-D's `--combine-runs` merges runs before conngraph sees the data; for XCP-D output without it, `--combine-runs` together with `--connectivity` z-scores each run's time series and concatenates them in run order before computing connectivity.
