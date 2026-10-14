#!/usr/bin/env bash
# Download, normalize, materialize and verify the frozen final benchmark.
set +x  # Never trace credentials, even when invoked with bash -x.
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
registry="$repo_root/data/pxrd_bench_registry"
cache="$repo_root/data/sources"
output="$repo_root/data/pxrd_bench"
oqmd_table=""
offline=0
shard_size=2048
workers=1
interval=1
verify_only=0
environment_only=0
bootstrap=1
stage="arguments"

usage() {
    cat <<'EOF'
Usage: bash data/setup_pxrd_bench.sh [options]
  --registry PATH      Frozen final registry (default: data/pxrd_bench_registry)
  --cache PATH         Local upstream downloads/index (default: data/sources)
  --output PATH        Dataset destination (default: data/pxrd_bench)
  --oqmd-table PATH    Existing verified OQMD v1.5 exported CSV.gz
  --offline            Reuse only local sources; never contact providers
  --shard-size N       Rows per new tensor shard (default: 2048)
  --workers N          CPU workers for OQMD export/reconstruction (default: 1)
  --interval SECONDS   Delay after API calls; at least 1 second
  --verify-only        Recheck every output entry without downloading/building
  --environment-only   Prepare the CPU Python environment, then exit
  --no-bootstrap       Require an already prepared Python/MariaDB environment
  --help               Show this message

Automatically prepares Python 3.12 and frozen CPU packages in .venv; missing
MariaDB tools are downloaded locally for OQMD. No sudo or system service changes.
MP downloads use MP_API_KEY (hidden terminal prompt if unset). Re-running verifies
a completed output or resumes a partial build. --offline forbids ALL downloads.
Source terms apply to all downloads and locally generated data.
EOF
}
while (($#)); do
    case "$1" in
        --registry|--cache|--output|--oqmd-table|--shard-size|--workers|--interval)
            (($# >= 2)) || { printf 'Missing value: %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --registry) registry="$2";; --cache) cache="$2";; --output) output="$2";;
                --oqmd-table) oqmd_table="$2";; --shard-size) shard_size="$2";; --workers) workers="$2";; --interval) interval="$2";;
            esac
            shift 2;;
        --offline) offline=1; shift;;
        --verify-only) verify_only=1; shift;;
        --environment-only) environment_only=1; shift;;
        --no-bootstrap) bootstrap=0; shift;;
        --help|-h) usage; exit 0;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2;;
    esac
done
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { printf 'Workers must be a positive integer.\n' >&2; exit 2; }
[[ "$shard_size" =~ ^[1-9][0-9]*$ ]] || { printf 'Shard size must be a positive integer.\n' >&2; exit 2; }
[[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk "BEGIN {exit !($interval >= 1)}" || {
    printf 'Interval must be at least one second.\n' >&2; exit 2;
}
if ((!environment_only)); then
    [[ -f "$registry/manifest.json" ]] || { printf 'Missing frozen registry: %s/manifest.json\n' "$registry" >&2; exit 1; }
fi
mkdir -p -- "$cache"
cache="$(cd -- "$cache" && pwd)"
# Serialize environment changes and builds in this checkout when flock is available.
if command -v flock >/dev/null; then
    exec 9> "$repo_root/data/.pxrd-bench-setup.lock"
    flock -n 9 || { printf 'Another PXRD-Bench setup is using this checkout.\n' >&2; exit 1; }
fi
mkdir -p -- "$cache/logs"
log="$cache/logs/setup-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
exec > >(tee -a "$log") 2>&1
printf 'PXRD-Bench setup log: %s\n' "$log"
trap 'code=$?; printf "Setup stopped during %s (exit %s). Log: %s\n" "$stage" "$code" "$log" >&2' ERR

tools="$cache/tools"
mkdir -p -- "$tools"
# Only official HTTPS binaries with a fixed SHA-256 are unpacked.
download_tool() {
    local url="$1" destination="$2" expected="$3"
    if [[ ! -f "$destination" ]]; then
        ((!offline)) || { printf 'Offline tool missing: %s\n' "$destination" >&2; return 1; }
        command -v curl >/dev/null || { printf 'Install curl to enable automatic local tools.\n' >&2; return 1; }
        curl --fail --location --proto '=https' --proto-redir '=https' --retry 3 --connect-timeout 30 --speed-time 120 --speed-limit 1024 \
            --continue-at - --output "$destination.part" "$url"
        printf '%s  %s\n' "$expected" "$destination.part" | sha256sum --check --status
        mv -- "$destination.part" "$destination"
    fi
    printf '%s  %s\n' "$expected" "$destination" | sha256sum --check --status
}
prepare_uv() {
    uv_bin="$tools/uv-0.9.0/uv"
    if [[ ! -x "$uv_bin" ]]; then
        command -v tar >/dev/null || { printf 'Install tar to enable local tools.\n' >&2; return 1; }
        download_tool 'https://github.com/astral-sh/uv/releases/download/0.9.0/uv-x86_64-unknown-linux-gnu.tar.gz' \
            "$tools/uv-0.9.0.tar.gz" '4dadaa5ff5009ccd6a0a43f6ccfa32bf36ed2eff18df7011275a9b1d81950e7b'
        local unpack
        unpack="$(mktemp -d "$tools/uv-unpack.XXXXXXXX")"
        tar -xzf "$tools/uv-0.9.0.tar.gz" -C "$unpack" --strip-components=1
        mv -- "$unpack" "$tools/uv-0.9.0"
    fi
    export UV_PYTHON_INSTALL_DIR="$tools/python"
    export UV_CACHE_DIR="$tools/uv-cache"
    uv_args=(--no-config)
    if ((offline)); then uv_args+=(--offline); fi
}
prepare_environment() {
    stage="Python environment"
    if [[ -x .venv/bin/python ]] && .venv/bin/python -m src.pxrd_bench_setup.environment --quiet; then
        printf '[environment] Frozen Python dependencies already ready.\n'
        return
    fi
    ((bootstrap)) || { printf 'Prepare .venv or rerun without --no-bootstrap.\n' >&2; return 1; }
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
        printf 'Automatic reconstruction setup requires Linux x86-64.\n' >&2; return 1;
    }
    if [[ -e .venv ]]; then
        [[ -x .venv/bin/python ]] || { printf 'Existing .venv is incomplete; move it aside and rerun.\n' >&2; return 1; }
        .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Move the existing non-3.12 .venv aside before setup"'
    fi
    prepare_uv
    if [[ ! -x .venv/bin/python ]]; then
        printf '[environment] Creating local Python 3.12 environment.\n'
        "$uv_bin" "${uv_args[@]}" venv --python 3.12 .venv
    fi
    printf '[environment] Installing frozen scientific packages and CPU-only PyTorch.\n'
    "$uv_bin" "${uv_args[@]}" pip install --python .venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
    "$uv_bin" "${uv_args[@]}" pip install --python .venv/bin/python \
        --index-url https://pypi.org/simple -r data/requirements-reconstruction.txt
    .venv/bin/python -m src.pxrd_bench_setup.environment
}
prepare_mariadb() {
    local missing=0 program
    for program in mariadb-install-db mariadbd mariadb mariadb-admin; do
        command -v "$program" >/dev/null || missing=1
    done
    if ((missing)); then
        ((bootstrap)) || { printf 'Install MariaDB tools or rerun without --no-bootstrap.\n' >&2; return 1; }
        stage="local MariaDB tools"
        mariadb_home="$tools/mariadb-11.4.5"
        if [[ ! -x "$mariadb_home/bin/mariadbd" ]]; then
            download_tool 'https://archive.mariadb.org/mariadb-11.4.5/bintar-linux-systemd-x86_64/mariadb-11.4.5-linux-systemd-x86_64.tar.gz' \
                "$tools/mariadb-11.4.5.tar.gz" '2a44cb70a87dba7eb2cab3b5af2c0416a0204d93a8fda387b4b70c9f1bab7bd6'
            local unpack
            unpack="$(mktemp -d "$tools/mariadb-unpack.XXXXXXXX")"
            tar -xzf "$tools/mariadb-11.4.5.tar.gz" -C "$unpack" --strip-components=1
            mv -- "$unpack" "$mariadb_home"
        fi
        export PATH="$mariadb_home/bin:$mariadb_home/scripts:$PATH"
        database_args=(--basedir="$mariadb_home")
    fi
    for program in mariadbd mariadb mariadb-admin; do
        "$program" --no-defaults --version || {
            printf 'MariaDB cannot run on this system. Install its missing shared libraries or supply --oqmd-table.\n' >&2
            return 1
        }
    done
}
prepare_numeric() {
    stage="frozen canonical math libraries"
    export PXRD_BENCH_MATH_ROOT="$tools/canonical-math"
    local profile filename digest library_hash archive
    for profile in 2.31 2.35; do
        if [[ "$profile" == 2.31 ]]; then
            filename='libc6_2.31-0ubuntu9.18_amd64.deb'
            digest='1b2281aac4935dfea1f89dfc19e445fdbb45303202af60679ccb8bf035f081a0'
            library_hash='a0e70f48f02a399e8dd37f861a99e226c285a9daf50a25d19fff7ccf18b97f9c'
        else
            filename='libc6_2.35-0ubuntu3.15_amd64.deb'
            digest='79e35256227e16a607c154cdeb8d76ff12d20e31de286ea7fd9ad3b96fe0452d'
            library_hash='3dd5511ae94785c9f921429b0f2b2f7aabb461b6f0e6de6dfdbef15f24bdfee6'
        fi
        if [[ ! -f "$PXRD_BENCH_MATH_ROOT/$profile/libm.so.6" ]]; then
            ((bootstrap)) || { printf 'Frozen math profile %s is missing; rerun without --no-bootstrap.\n' "$profile" >&2; return 1; }
            archive="$tools/$filename"
            download_tool "https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/$filename" "$archive" "$digest"
            .venv/bin/python -m src.pxrd_bench_setup.numeric --profile "$profile" \
                --archive "$archive" --destination "$PXRD_BENCH_MATH_ROOT/$profile"
        fi
        printf '%s  %s\n' "$library_hash" "$PXRD_BENCH_MATH_ROOT/$profile/libm.so.6" | sha256sum --check --status
    done
}
prepare_environment
if ((environment_only)); then printf 'CPU reconstruction environment ready: %s/.venv\n' "$repo_root"; exit 0; fi
# Match the original CPU math dispatch; SIMD/BLAS paths change float bytes.
if ((!verify_only)); then
    export NPY_DISABLE_CPU_FEATURES='AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL'
    export OPENBLAS_CORETYPE=Haswell
fi
common=(--registry "$registry" --cache "$cache" --output "$output")
if ((verify_only)) || [[ -e "$output" ]]; then
    stage="existing dataset verification"
    .venv/bin/python -m src.pxrd_bench_setup.reconstruct verify "${common[@]}"
    printf 'PXRD-Bench verification passed: %s\n' "$output"
    exit 0
fi
stage="registry and source prerequisites"
.venv/bin/python -m src.pxrd_bench_setup.reconstruct check "${common[@]}"
# Fail before large source downloads; credentials are never written to a file.
mp_needed="$(.venv/bin/python -m src.pxrd_bench_setup.reconstruct mp-info "${common[@]}")"
if [[ -n "$mp_needed" ]]; then
    ((!offline)) || { printf 'Offline MP records are missing; rerun online with MP_API_KEY.\n' >&2; exit 1; }
    if [[ -z "${MP_API_KEY:-}" ]]; then
        if [[ -t 0 ]]; then
            printf 'Materials Project API key (https://next-gen.materialsproject.org/api): ' >&2
            IFS= read -rs MP_API_KEY
            printf '\n' >&2
        fi
        [[ -n "${MP_API_KEY:-}" ]] || { printf 'MP_API_KEY is required. Set it in the environment or run from an interactive terminal.\n' >&2; exit 1; }
        export MP_API_KEY
    fi
fi
prepare_numeric

mysql_pid=""
mysql_tmp=""
database_args=()
mariadb_home=""
cleanup() {
    if [[ -n "$mysql_pid" ]]; then
        kill "$mysql_pid" 2>/dev/null || true
        wait "$mysql_pid" 2>/dev/null || true
    fi
    if [[ -n "$mysql_tmp" ]]; then
        rm -rf -- "$mysql_tmp"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "$oqmd_table" ]]; then
    plan="$cache/oqmd-plan.txt"
    .venv/bin/python -m src.pxrd_bench_setup.reconstruct oqmd-info "${common[@]}" > "$plan"
    if [[ -s "$plan" ]]; then
        mapfile -t oqmd_info < "$plan"
        [[ ${#oqmd_info[@]} == 3 ]] || { printf 'Invalid OQMD reconstruction plan.\n' >&2; exit 1; }
        oqmd_table="$cache/oqmd/export/oqmd_v1_5_auxiliary.csv.gz"
        if [[ ! -f "$oqmd_table" ]]; then
            command -v gzip >/dev/null || { printf 'Install gzip for the OQMD snapshot.\n' >&2; exit 1; }
            prepare_mariadb
            stage="OQMD snapshot download and import"
            download_args=()
            if ((offline)); then download_args+=(--offline); fi
            .venv/bin/python -m src.pxrd_bench_setup.reconstruct download-resource "${common[@]}" \
                --resource-source oqmd --resource-key "${oqmd_info[0]}" "${download_args[@]}"
            dump_path="${oqmd_info[1]}"
            oqmd_database="${oqmd_info[2]}"
            datadir="$cache/oqmd/mysql-data"
            mkdir -p -- "$datadir"
            datadir="$(cd -- "$datadir" && pwd)"
            if [[ ! -d "$datadir/mysql" ]]; then
                mariadb-install-db --no-defaults "${database_args[@]}" --datadir="$datadir" \
                    --auth-root-authentication-method=normal --skip-test-db > "$cache/oqmd/mysql-init.log" 2>&1
            fi
            mysql_tmp="$(mktemp -d /tmp/pxrd-bench-oqmd.XXXXXXXX)"
            mysql_socket="$mysql_tmp/mysql.sock"
            db_buffer_pool="$(.venv/bin/python -m src.pxrd_bench_setup.environment --oqmd-buffer-pool)"
            [[ "$db_buffer_pool" =~ ^[0-9]+$ ]] || { printf 'Invalid private DB buffer size.\n' >&2; exit 1; }
            printf '[OQMD] Private SQL buffer pool: %s bytes (available-memory and cgroup limits applied).\n' "$db_buffer_pool"
            mariadbd --no-defaults "${database_args[@]}" --user="$(id -un)" --datadir="$datadir" --socket="$mysql_socket" \
                --pid-file="$mysql_tmp/mysql.pid" --skip-networking --skip-name-resolve \
                --log-error="$datadir/server.log" --innodb-buffer-pool-size="$db_buffer_pool" --innodb-log-file-size=1G \
                --max-allowed-packet=1G &
            mysql_pid=$!
            ready=0
            for ((attempt=0; attempt<60; attempt++)); do
                if mariadb-admin --no-defaults --user=root --socket="$mysql_socket" ping >/dev/null 2>&1; then ready=1; break; fi
                kill -0 "$mysql_pid" 2>/dev/null || break
                sleep 1
            done
            ((ready)) || { printf 'Private OQMD database failed to start; see %s/server.log\n' "$datadir" >&2; exit 1; }
            import_marker="$datadir/import.complete"
            dump_hash="$(sha256sum -- "$dump_path")"
            dump_hash="${dump_hash%% *}"
            if [[ -f "$import_marker" && "$(cat -- "$import_marker")" != "$dump_hash" ]]; then
                printf 'OQMD datadir belongs to another snapshot. Choose a new --cache.\n' >&2; exit 1
            fi
            if [[ ! -f "$import_marker" ]]; then
                printf '[OQMD] Importing the complete verified SQL snapshot; this can take a long time.\n'
                # Restart an interrupted import only in our private snapshot database.
                mariadb --no-defaults --user=root --socket="$mysql_socket" \
                    -e "DROP DATABASE IF EXISTS \`$oqmd_database\`; CREATE DATABASE \`$oqmd_database\`;"
                gzip -dc -- "$dump_path" | mariadb --no-defaults --user=root --socket="$mysql_socket" "$oqmd_database"
                printf '%s\n' "$dump_hash" > "$import_marker"
                printf '[OQMD] Complete SQL import passed. Exporting source structures.\n'
            fi
            .venv/bin/python -m src.pxrd_bench_setup.oqmd \
                --mysql-bin "$(command -v mariadb)" --mysql-user root --socket "$mysql_socket" \
                --database "$oqmd_database" --dump-path "$dump_path" --output-dir "$cache/oqmd/export" --workers "$workers"
            cleanup
            mysql_pid=""; mysql_tmp=""
        fi
    fi
fi

stage="selected source download and normalization"
fetch_args=(--interval "$interval")
if ((offline)); then fetch_args+=(--offline); fi
if [[ -n "$oqmd_table" ]]; then fetch_args+=(--oqmd-table "$oqmd_table"); fi
.venv/bin/python -m src.pxrd_bench_setup.reconstruct fetch "${common[@]}" "${fetch_args[@]}"
stage="dataset reconstruction and full verification"
.venv/bin/python -m src.pxrd_bench_setup.reconstruct build "${common[@]}" --shard-size "$shard_size" --workers "$workers"
printf 'PXRD-Bench reconstruction and full content verification passed: %s\n' "$output"
