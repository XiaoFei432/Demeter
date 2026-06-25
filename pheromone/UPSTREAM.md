# Pheromone Upstream

This directory vendors the Pheromone source tree as ordinary repository content.
It is intentionally not a Git submodule and must not contain a nested `.git/`
directory when committed to the Demeter repository.

- Upstream repository: https://github.com/MincYu/pheromone.git
- Vendored upstream commit: `4d22677aa20dd94069da06d47ec2608504bf6bf4`
- Original cloned remote in this workspace: `https://ghproxy.net/https://github.com/MincYu/pheromone.git`

The vendored tree included local Demeter integration changes when it was folded
into this repository.

Files modified relative to the upstream commit:

- `client/pheromone/client.py`
- `common/proto/operation.proto`
- `dockerfiles/start-pheromone.sh`
- `include/coordinator/coord_handlers.hpp`
- `include/scheduler/comm_helper.hpp`
- `src/coordinator/CMakeLists.txt`
- `src/coordinator/coordinator.cpp`
- `src/coordinator/func_call_handler.cpp`
- `src/executor/executor_server.cpp`
- `src/scheduler/scheduler_server.cpp`

Files added for Demeter integration:

- `conf/demeter-example.yml`
- `include/coordinator/demeter_scheduler.hpp`
- `src/coordinator/demeter_scheduler.cpp`

To refresh this directory from upstream, sync the external Pheromone repository
outside this checkout, copy the desired files into `pheromone/`, then remove any
nested `.git/` metadata before committing.
