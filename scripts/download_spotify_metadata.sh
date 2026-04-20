#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_DATA_DIR="${PROJECT_ROOT}/data"
DATASET_NAME="spotify-metadata"
DATASET_LINK="${REPO_DATA_DIR}/${DATASET_NAME}"

# Modo full (Kaggle)
KAGGLE_DATASET="lordpatil/spotify-metadata-by-annas-archive"
KAGGLE_CONFIG="${HOME}/.kaggle/kaggle.json"

# Modo truncated (GitHub release do próprio repo)
TRUNCATED_ZIP_NAME="spotify-metadata-by-annas-archive-truncated-300mb.zip"
RELEASE_TAG="v0.1-data"
REPO_SLUG="carsio/music-search-engine"

MODE="truncated"
DESTINATION_INPUT=""

usage() {
    cat <<EOF
Uso: $(basename "$0") [--truncated | --full] [DESTINO]

Baixa o dataset ${DATASET_NAME} e extrai em data/${DATASET_NAME}.

MODOS:
  --truncated  Subset de ~344 MB do GitHub release ${RELEASE_TAG} (padrão, rápido).
               Fast path: se data/${TRUNCATED_ZIP_NAME} já existir, pula o download.
  --full       Dataset completo (~5.5 GB) via Kaggle CLI.

DESTINO (opcional):
  - omitido: extrai em data/${DATASET_NAME}
  - caminho: extrai em <destino>/${DATASET_NAME} e cria symlink em data/
  - caminho terminando em /${DATASET_NAME}: usa exatamente esse diretório

O layout final é idêntico nos dois modos, então notebooks e código de
indexação não precisam saber qual foi baixado.
EOF
}

resolve_from_project_root() {
    local path="$1"

    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${PROJECT_ROOT}/${path}"
    fi
}

normalize_path() {
    local path="$1"
    local parent_dir
    local base_name

    parent_dir="$(dirname "${path}")"
    base_name="$(basename "${path}")"
    mkdir -p "${parent_dir}"

    (
        cd "${parent_dir}"
        printf '%s/%s\n' "$(pwd -P)" "${base_name}"
    )
}

target_dir_from_input() {
    local input="$1"
    local candidate

    candidate="$(resolve_from_project_root "${input}")"
    candidate="${candidate%/}"

    if [[ -z "${candidate}" ]]; then
        candidate="/"
    fi

    if [[ "$(basename "${candidate}")" == "${DATASET_NAME}" ]]; then
        printf '%s\n' "${candidate}"
    else
        printf '%s/%s\n' "${candidate}" "${DATASET_NAME}"
    fi
}

log() {
    printf '[%s] %s\n' "${MODE}" "$*"
}

fetch_truncated() {
    local target_dir="$1"
    local zip_path="${REPO_DATA_DIR}/${TRUNCATED_ZIP_NAME}"

    if [[ -f "${zip_path}" ]]; then
        log "zip local encontrado em ${zip_path} — pulando download"
    elif command -v gh >/dev/null 2>&1; then
        log "baixando asset ${TRUNCATED_ZIP_NAME} da release ${RELEASE_TAG} via gh"
        gh release download "${RELEASE_TAG}" \
            --repo "${REPO_SLUG}" \
            --pattern "${TRUNCATED_ZIP_NAME}" \
            -D "${REPO_DATA_DIR}"
    elif command -v curl >/dev/null 2>&1; then
        local url="https://github.com/${REPO_SLUG}/releases/download/${RELEASE_TAG}/${TRUNCATED_ZIP_NAME}"
        log "baixando ${url} via curl"
        curl -L --fail -o "${zip_path}" "${url}"
    else
        echo "[${MODE}] Erro: instale 'gh' ou 'curl' para baixar o dataset." >&2
        exit 1
    fi

    log "extraindo em ${target_dir}"
    unzip -q -o "${zip_path}" -d "${target_dir}"
}

fetch_full() {
    local target_dir="$1"

    if ! command -v uv >/dev/null 2>&1; then
        echo "[${MODE}] Erro: 'uv' não está instalado ou não está no PATH." >&2
        exit 1
    fi

    if [[ ! -f "${KAGGLE_CONFIG}" ]]; then
        echo "[${MODE}] Erro: credencial do Kaggle não encontrada em ${KAGGLE_CONFIG}." >&2
        echo "[${MODE}] Crie o token em Kaggle > Settings > Create New Token." >&2
        exit 1
    fi

    chmod 600 "${KAGGLE_CONFIG}"
    log "baixando ${KAGGLE_DATASET} do Kaggle em ${target_dir}"
    uv run kaggle datasets download \
        -d "${KAGGLE_DATASET}" \
        -p "${target_dir}" \
        --unzip
}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --truncated)
            MODE="truncated"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --)
            shift
            DESTINATION_INPUT="${1:-}"
            shift || true
            ;;
        -*)
            echo "Erro: flag desconhecida '$1'." >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -n "${DESTINATION_INPUT}" ]]; then
                echo "Erro: use no máximo um destino posicional." >&2
                usage >&2
                exit 1
            fi
            DESTINATION_INPUT="$1"
            shift
            ;;
    esac
done

if [[ -z "${DESTINATION_INPUT}" ]]; then
    DESTINATION_INPUT="data"
fi

TARGET_DATASET_DIR="$(target_dir_from_input "${DESTINATION_INPUT}")"
TARGET_DATASET_DIR="$(normalize_path "${TARGET_DATASET_DIR}")"

mkdir -p "${REPO_DATA_DIR}"

if [[ "${TARGET_DATASET_DIR}" == "${DATASET_LINK}" ]]; then
    if [[ -L "${DATASET_LINK}" ]]; then
        rm -f "${DATASET_LINK}"
    elif [[ -e "${DATASET_LINK}" && ! -d "${DATASET_LINK}" ]]; then
        echo "[${MODE}] Erro: ${DATASET_LINK} existe e não pode ser usado como diretório." >&2
        exit 1
    fi
fi

mkdir -p "${TARGET_DATASET_DIR}"

case "${MODE}" in
    truncated) fetch_truncated "${TARGET_DATASET_DIR}" ;;
    full)      fetch_full "${TARGET_DATASET_DIR}" ;;
esac

if [[ "${TARGET_DATASET_DIR}" != "${DATASET_LINK}" ]]; then
    if [[ -L "${DATASET_LINK}" || ! -e "${DATASET_LINK}" ]]; then
        rm -f "${DATASET_LINK}"
    elif [[ -d "${DATASET_LINK}" ]]; then
        if find "${DATASET_LINK}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            echo "[${MODE}] Erro: ${DATASET_LINK} já existe como diretório não vazio." >&2
            echo "[${MODE}] Remova ou mova o conteúdo antes de recriar o symlink." >&2
            exit 1
        fi
        rmdir "${DATASET_LINK}"
    else
        echo "[${MODE}] Erro: ${DATASET_LINK} existe e não pode ser substituído automaticamente." >&2
        exit 1
    fi

    ln -s "${TARGET_DATASET_DIR}" "${DATASET_LINK}"
    log "dataset disponível em ${DATASET_LINK} (origem: ${TARGET_DATASET_DIR})"
else
    log "dataset disponível em ${DATASET_LINK}"
fi
