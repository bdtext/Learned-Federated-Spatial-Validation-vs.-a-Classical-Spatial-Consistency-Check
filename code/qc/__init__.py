"""Quality-control benchmark library (written from scratch, October 2026).
Modules: data (networks, climatology, splits, audits), inject (seeded fault injection + audit), detectors (classical and
no-training checks, scoring, thresholds), model (leave-self-out spatiotemporal graph validator, PyTorch), fl (federated
training, aggregation rules, attacks), metrics (point, event, bootstrap, seed-level tests, TOST), govsim (verification by
re-computation, simulated).
"""
