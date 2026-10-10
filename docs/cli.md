# CLI reference

```text
conngraph INPUT OUTPUT {participant,group} [options]
```

The tables on this page are generated from the command itself, so they match `conngraph --help`.

```{include} _generated/general.md
```

## Input options

Give the same input options at both levels. The [user guide](guide.md#input) describes the input types and the folder layout.

```{include} _generated/input.md
```

## Options for both levels

```{include} _generated/both-levels.md
```

## Participant level options

These options are accepted only at the participant level; `--graph-method` is required there.

```{include} _generated/participant.md
```

## Group level options

These options are accepted only at the group level. `--groups` turns on the comparisons and needs `--group-column` and `--contrast`; `--correction` is required too, unless an empty `--compare` leaves only NBS. [Comparison tables](guide.md#comparison-tables) describes the result tables.

```{include} _generated/group.md
```
