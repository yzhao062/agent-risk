# Pilot sandbox: ControlArena's bash image (pre-built as bashbuild-default by
# bootstrap_vm.sh) with a dedicated task working directory. Tests use relative
# paths (./solution.sh, test_task.py), so the working directory only moves.
FROM bashbuild-default:latest
RUN mkdir -p /workspace
WORKDIR /workspace
