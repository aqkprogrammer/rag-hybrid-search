---
title: Deployment & Rollback Runbook
department: Engineering
owner: Platform Team
classification: internal
version: "2.3"
last_reviewed: 2026-04-02
---

# Deployment & Rollback Runbook

All production services at Lumora are deployed through the Launchpad pipeline, which builds
container images, runs tests and rolls changes out to Kubernetes.

## Deployment Windows

Production deployments are allowed Monday to Thursday between 09:00 and 16:00 UTC. Deployments
on Fridays, weekends and public holidays require approval from the on-call Engineering Manager.
A company-wide release freeze applies during the last two weeks of December.

## Pre-deployment Checklist

- The change has been reviewed and approved by at least one other engineer.
- All CI checks, including integration tests, are green.
- Database migrations are backwards compatible with the currently deployed version.
- A feature flag guards any user-facing behaviour change.

## Progressive Rollout

Launchpad uses canary releases. A new version first receives 5% of traffic for 15 minutes. If the
error rate stays below 1% and p99 latency does not regress by more than 10%, traffic is increased
to 25%, then 50%, then 100%. The canary analysis is automatic; a failed check halts the rollout.

## Rolling Back

To roll back a service, run:

```bash
launchpad rollback <service> --to previous
```

A rollback restores the previous container image and configuration and normally completes in
under 3 minutes. Database migrations are never rolled back automatically; use a forward fix.

Roll back immediately, without waiting for a root cause, if a deployment causes a SEV1 or SEV2
incident.

## Hotfixes

Hotfixes follow the same pipeline but may skip the 15-minute canary stage with approval from the
Incident Commander during an active incident.
