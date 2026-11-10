# meridian-cli Reference

Install with `brew install northwind/tap/meridian` or download the binary from
the releases page. The CLI reads credentials from `~/.meridian/config.toml`.

## Common commands

`meridian stack list` — show every Stack you have access to in the current Lane.

`meridian deploy --stack NAME` — build a new Envelope from the current directory
and deploy it. Add `--dry-run` to see the diff without applying it.

`meridian rollback --stack NAME --to ENVELOPE_ID` — point a Stack at an older
Envelope. This takes effect in under 30 seconds because nothing is rebuilt.

`meridian promote --stack NAME --from staging --to prod` — copy a sealed
Envelope from one Lane to another. Promotion to `prod` requires a second
approver unless the Stack is marked `low_risk` in its manifest.

`meridian logs --stack NAME --follow` — stream logs. Defaults to the last 15
minutes; use `--since 2h` to widen the window.

`meridian secrets set --stack NAME KEY=VALUE` — write a secret. Secrets are
encrypted with the Lane's KMS key and are never printed back by the CLI.

## Flags worth knowing

`--lane` overrides the Lane for a single command. Without it the CLI uses the
Lane in your config file, which defaults to `dev`.

`--json` makes any command emit machine-readable output. Use this in scripts
rather than parsing the human-readable tables, which are not stable.

## Exit codes

- `0` — success
- `2` — validation failure in the manifest
- `3` — the controller rejected the deploy (usually a quota problem)
- `4` — authentication failed; run `meridian auth login`
