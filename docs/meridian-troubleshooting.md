# Meridian — Troubleshooting

## Deploy hangs at "waiting for controller"

The controller batches reconciliation every 20 seconds, so a short wait is
normal. If it exceeds two minutes, the controller is usually backed up behind a
large Stack. Check `meridian status --controller`. There is nothing to fix on
your side; the deploy will proceed.

## Exit code 3 on deploy

The Lane is out of quota. Each Lane has a CPU and memory budget set per team.
Run `meridian quota show` to see current usage. Either scale down an unused
Stack or request an increase in #platform-help. Quota increases are approved by
the platform team and usually land the same day.

## Rollback did not change anything

Rollback only swaps the Envelope. If the problem is in data — a bad migration,
a corrupted cache — rolling back the Envelope will not help. Migrations are not
reversed automatically; you must write a down-migration and deploy it forward.

## Secrets appear empty inside the container

Secrets are injected at container start, not at build time. If you added a
secret after the last deploy, the running container will not have it. Redeploy
or restart the Stack with `meridian restart --stack NAME`.

## "Envelope is sealed" error when editing config

Envelopes are immutable by design. Edit the manifest in your working directory
and run `meridian deploy` to create a new one. There is no way to mutate a
sealed Envelope, and this is deliberate — it is what makes rollback reliable.

## Authentication keeps expiring

Tokens last eight hours. If you are being logged out more often than that, your
clock is probably skewed; the controller rejects tokens with more than 60
seconds of drift. Enable NTP sync.
