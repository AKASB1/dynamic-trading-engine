"""Canaries of the information audit: strategies that leak on purpose. Each module exposes
CANARY (id, name, the guard that must catch it) and a Strategy class. Modules whose names
start with an underscore are helpers and are skipped by the canary loader."""
