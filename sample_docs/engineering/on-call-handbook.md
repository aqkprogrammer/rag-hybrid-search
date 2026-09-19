---
title: On-Call Handbook
department: Engineering
owner: SRE Team
classification: internal
version: "1.6"
last_reviewed: 2026-02-10
---

# On-Call Handbook

## Rotation

Each product team runs its own weekly on-call rotation. Shifts start on Monday at 10:00 local
time and last one week. Every rotation has a primary and a secondary engineer. New engineers
shadow two full rotations before joining as secondary.

## Paging Expectations

The primary on-call engineer must acknowledge a page within 5 minutes and be at a laptop within
15 minutes. If the primary does not acknowledge within 10 minutes, PagerDuty escalates to the
secondary, and then to the Engineering Manager after 20 minutes.

## Compensation

Engineers receive an on-call stipend of $500 per week of primary on-call and $250 per week as
secondary. Engineers who are paged between 22:00 and 07:00 may start work late the next day or
take a half day off in lieu.

## Handoff

At the end of a shift, the outgoing primary writes a handoff note in the team channel listing
open incidents, noisy alerts and any in-progress mitigations.

## Alert Hygiene

Every page must be actionable. Alerts that fire more than 3 times in a week without requiring
action must be tuned or deleted in the following sprint.
