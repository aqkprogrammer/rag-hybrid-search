---
title: Incident Response Runbook
department: Engineering
owner: SRE Team
classification: internal
version: "4.1"
last_reviewed: 2026-05-20
---

# Incident Response Runbook

This runbook defines how Lumora engineering detects, declares, mitigates and learns from
production incidents.

## Severity Levels

| Severity | Definition | Response target |
|---|---|---|
| SEV1 | Full outage or data loss affecting all customers | Acknowledge in 5 minutes |
| SEV2 | Major feature degraded for many customers | Acknowledge in 15 minutes |
| SEV3 | Minor degradation, workaround available | Acknowledge in 1 business day |

Any engineer may declare an incident. When in doubt, declare it at a higher severity and
downgrade later.

## Declaring an Incident

1. Run `/incident declare` in Slack. The bot creates a dedicated channel named
   `#inc-YYYYMMDD-short-name` and pages the on-call Incident Commander.
2. Post a one-line summary of customer impact in the channel.
3. Open the incident in PagerDuty if it was not triggered by an alert.

## Roles

### Incident Commander

The Incident Commander (IC) coordinates the response, makes decisions and delegates work. The
IC does not debug or fix the problem themselves. For SEV1 incidents the IC must post status
updates in the incident channel every 30 minutes.

### Communications Lead

For SEV1 and SEV2 incidents a Communications Lead updates the public status page within 20
minutes of declaration and every hour afterwards until resolution.

### Subject Matter Experts

Engineers from the affected services investigate and implement mitigations, reporting progress
to the IC.

## Mitigation First

Restore service before finding the root cause. Preferred mitigations, in order: roll back the
most recent deployment, disable the feature flag, fail over to the secondary region, scale out.

## Postmortems

A blameless postmortem is required for every SEV1 and SEV2 incident. The draft must be
published within 5 business days of resolution and reviewed at the weekly reliability review.
Every postmortem action item must have an owner and a due date in Jira.
