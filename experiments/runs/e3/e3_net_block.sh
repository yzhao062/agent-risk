#!/usr/bin/env bash
# E3 egress block (host rule). E3 has no sandbox containers: the episode process itself
# (tau2 environment, user simulator client, monitor client) is what must not reach any
# other host. The runner runs with the dedicated group e3net (sg e3net), and one
# iptables OUTPUT rule rejects every packet from processes of that group that does not
# leave through the loopback interface. The vLLM servers (loopback only) and sshd are
# not in the group, so serving and administration are unaffected. Idempotent.
#   bash e3_net_block.sh            # create the group, install the rules
#   bash e3_net_block.sh --check    # exit 0 iff both rules are present
#   bash e3_net_block.sh --status   # rules with counters
#   bash e3_net_block.sh --remove
set -uo pipefail
G=e3net
R1=(-m owner --gid-owner "$G" ! -o lo -p tcp -j REJECT --reject-with tcp-reset)
R2=(-m owner --gid-owner "$G" ! -o lo -j REJECT --reject-with icmp-port-unreachable)
TARGET_USER=${E3_USER:-${USER:-ubuntu}}
case "${1:-}" in
  --check)
    sudo -n iptables -C OUTPUT "${R1[@]}" 2>/dev/null && sudo -n iptables -C OUTPUT "${R2[@]}" 2>/dev/null; exit $?;;
  --status)
    sudo iptables -L OUTPUT -v -x -n | head -12; getent group "$G"; exit 0;;
  --remove)
    sudo iptables -D OUTPUT "${R1[@]}" 2>/dev/null; sudo iptables -D OUTPUT "${R2[@]}" 2>/dev/null; echo removed; exit 0;;
esac
getent group "$G" >/dev/null || sudo groupadd "$G"
id -nG "$TARGET_USER" 2>/dev/null | grep -qw "$G" || sudo usermod -aG "$G" "$TARGET_USER" 2>/dev/null || true
sudo iptables -C OUTPUT "${R2[@]}" 2>/dev/null || sudo iptables -I OUTPUT 1 "${R2[@]}"
sudo iptables -C OUTPUT "${R1[@]}" 2>/dev/null || sudo iptables -I OUTPUT 1 "${R1[@]}"
sudo iptables -L OUTPUT -v -x -n | head -8
