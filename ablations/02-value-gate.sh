set -ex
# Value-weight gate: thin wrapper around the canonical ../run-valuegate.sh
# (imag_value_weight=0.05 with the PER fix; actor 0.1; entropy pinned to the
# pre-floor behavior so the gate stays single-variable vs the c6e22ae
# reference seeds).
#
# Arms via env vars, e.g.:
#   bash ablations/02-value-gate.sh                    # VW=0.05, RUN=13
#   VW=0 RUN=14 bash ablations/02-value-gate.sh        # value-off control
#
# Red-flag bands (kill early if TD Error/DQNLoss diverge in first 20k grad
# steps): Kangaroo <1000, Hero <5000, Jamesbond <600, Pong <14.
# ChopperCommand >=7000 would mean the value signal genuinely helps there.
exec bash "$(dirname "$0")/../run-valuegate.sh"
