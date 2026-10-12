#!/usr/bin/env bash
# Block all network egress from docker bridge networks (the pilot sandboxes), on
# the host. Idempotent. Packets still leave the sandbox interface, so the
# sandbox's tx counter records the attempt; the host then rejects them:
#   FORWARD (via DOCKER-USER): bridge -> anything that is not a bridge
#   INPUT: bridge -> the host itself (sshd etc.)
# TCP gets a reset, everything else an ICMP port-unreachable, so attempts fail
# fast instead of hanging.
#   bash net_block.sh          # install
#   bash net_block.sh --status # show rules and counters
#   bash net_block.sh --remove # remove
set -uo pipefail
rules_fwd=(
  "-i br-+ ! -o br-+ -p tcp -j REJECT --reject-with tcp-reset"
  "-i br-+ ! -o br-+ -j REJECT --reject-with icmp-port-unreachable"
)
rules_in=(
  "-i br-+ -p tcp -j REJECT --reject-with tcp-reset"
  "-i br-+ -j REJECT --reject-with icmp-port-unreachable"
)
case "${1:-}" in
  --status)
    sudo iptables -L DOCKER-USER -v -x -n; sudo iptables -L INPUT -v -x -n | head -20; exit 0;;
  --remove)
    for r in "${rules_fwd[@]}"; do sudo iptables -D DOCKER-USER $r 2>/dev/null; done
    for r in "${rules_in[@]}"; do sudo iptables -D INPUT $r 2>/dev/null; done
    echo removed; exit 0;;
esac
# insert in reverse so the order in the chain is as listed
for (( i=${#rules_fwd[@]}-1; i>=0; i-- )); do
  sudo iptables -C DOCKER-USER ${rules_fwd[$i]} 2>/dev/null || sudo iptables -I DOCKER-USER 1 ${rules_fwd[$i]}
done
for (( i=${#rules_in[@]}-1; i>=0; i-- )); do
  sudo iptables -C INPUT ${rules_in[$i]} 2>/dev/null || sudo iptables -I INPUT 1 ${rules_in[$i]}
done
sudo iptables -L DOCKER-USER -v -x -n
