---
title: Information Security Policy
department: Security
owner: CISO Office
classification: internal
version: "5.0"
last_reviewed: 2026-06-01
---

# Information Security Policy

This policy sets the minimum security requirements for all Lumora employees, contractors and
systems. Violations may result in disciplinary action.

## Data Classification

Lumora classifies information into four levels:

- **Public**: approved for public release, such as marketing material.
- **Internal**: default level for company information; may be shared with all employees.
- **Confidential**: customer data, financial results and source code; shared on a need-to-know basis.
- **Restricted**: credentials, encryption keys and personal health or payment data; access is logged
  and limited to named individuals.

Restricted data must never be stored in chat tools, email or personal cloud storage.

## Passwords and Authentication

Passwords must be at least 14 characters long and must be stored in the company password
manager, 1Password. Password reuse across services is prohibited. Multi-factor authentication
(MFA) is mandatory for all company systems; hardware security keys are required for production
and administrative access.

## Device Security

All company laptops must use full-disk encryption, have automatic screen lock after 5 minutes
of inactivity and run the endpoint detection agent. Operating system security updates must be
installed within 14 days of release, or within 72 hours for critical vulnerabilities.

## Reporting Security Incidents

Suspected security incidents, including phishing emails, lost devices and accidental data
exposure, must be reported to security@lumora.example or the #security-help Slack channel within
1 hour of discovery. Lost or stolen devices must also be reported to IT so they can be remotely
wiped.

## Security Training

All employees complete security awareness training during onboarding and annually thereafter.
Engineers additionally complete secure coding training every year.

## Third-Party Software

New SaaS tools that will process Confidential or Restricted data require a security review
before purchase. The review usually takes 10 business days.
