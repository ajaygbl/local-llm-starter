# Meridian — Overview

Meridian is Northwind Labs' internal deployment platform. It was released in
March 2024 to replace the older Halyard scripts.

Meridian has three parts:

- **meridian-cli** — the command-line tool engineers use day to day.
- **meridian-controller** — the service that reconciles desired state against
  what is actually running in the cluster.
- **Meridian Console** — the web UI at console.meridian.internal.

## Core concepts

A **Stack** is the unit of deployment. One Stack maps to one service and owns
its own database credentials, secrets and scaling policy.

An **Envelope** is a versioned bundle of configuration. Every deploy creates a
new Envelope, and Envelopes are immutable once sealed. Rolling back means
pointing a Stack at an older Envelope — it never rebuilds anything.

A **Lane** is an environment: `dev`, `staging` or `prod`. A Stack exists
separately in each Lane, and promotion between Lanes copies the Envelope
rather than redeploying from source.

## Release cadence

Meridian ships every second Tuesday. Patch releases go out as needed and do not
require a change ticket. Minor releases require sign-off from the platform
on-call engineer.
