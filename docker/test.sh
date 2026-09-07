#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

# An isolated project keeps test containers and synthetic data separate from the app.
test_project="bowatt-test-$$"
compose() {
    docker compose -p "$test_project" -f compose.yaml -f docker/compose.test.yaml "$@"
}
cleanup() {
    test_status=$?
    trap - EXIT
    if [ "$test_status" -ne 0 ]; then
        compose logs --no-color --tail 100 backend frontend || true
    fi
    if ! compose down --volumes --remove-orphans; then
        test_status=1
    fi
    exit "$test_status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

compose up --build --wait backend frontend
compose run --rm --no-deps smoke-tests
compose up --force-recreate --wait backend
compose run --rm --no-deps smoke-tests python smoke_test.py --verify-persistence
