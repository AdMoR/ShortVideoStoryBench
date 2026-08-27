#!/usr/bin/env bash
#
# Build the agent jail. UID/GID match the invoking user so the files the agent
# writes into its mounted workspace come back owned by you.
#
#   docker/build.sh                 # veb-pi:latest, pi pinned by the Dockerfile
#   PI_VERSION=0.84.3 docker/build.sh
#
# Bumping PI_VERSION changes the CLI contract PiGenerator.build_argv is written
# against — run the test suite after, not just the build.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="${VEB_PI_IMAGE:-veb-pi:latest}"

args=(
	--file "${here}/Dockerfile.pi"
	--tag "${image}"
	--build-arg "UID=$(id -u)"
	--build-arg "GID=$(id -g)"
)
[[ -n "${PI_VERSION:-}" ]] && args+=(--build-arg "PI_VERSION=${PI_VERSION}")

docker build "${args[@]}" "${here}"
echo "built ${image}"
