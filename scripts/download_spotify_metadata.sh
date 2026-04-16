#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_DATA_DIR="${PROJECT_ROOT}/data"
DATASET_NAME="spotify-metadata"
DATASET_LINK="${REPO_DATA_DIR}/${DATASET_NAME}"
KAGGLE_DATASET="lordpatil/spotify-metadata-by-annas-archive"
KAGGLE_CONFIG="${HOME}/.kaggle/kaggle.json"

usage() {
    cat <<EOF
Uso: $(basename "$0") [DESTINO]

Baixa o dataset ${DATASET_NAME} do Kaggle.

DESTINO:
  - omitido ou "data": baixa em data/${DATASET_NAME}
  - diretório base: <destino>/${DATASET_NAME}
  - diretório final: <destino> (se terminar com /${DATASET_NAME})

Se o destino ficar fora de data/, o script recria data/${DATASET_NAME}
como symlink para preservar o layout esperado pelo projeto.
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

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -gt 1 ]]; then
    echo "Erro: use no maximo um argumento para definir o destino." >&2
    usage >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Erro: 'uv' nao esta instalado ou nao esta no PATH." >&2
    exit 1
fi

if [[ ! -f "${KAGGLE_CONFIG}" ]]; then
    echo "Erro: credencial do Kaggle nao encontrada em ${KAGGLE_CONFIG}." >&2
    echo "Crie o token em Kaggle > Settings > Create New Token." >&2
    exit 1
fi

DESTINATION_INPUT="${1:-data}"
TARGET_DATASET_DIR="$(target_dir_from_input "${DESTINATION_INPUT}")"
TARGET_DATASET_DIR="$(normalize_path "${TARGET_DATASET_DIR}")"

mkdir -p "${REPO_DATA_DIR}"

if [[ "${TARGET_DATASET_DIR}" == "${DATASET_LINK}" ]]; then
    if [[ -L "${DATASET_LINK}" ]]; then
        rm -f "${DATASET_LINK}"
    elif [[ -e "${DATASET_LINK}" && ! -d "${DATASET_LINK}" ]]; then
        echo "Erro: ${DATASET_LINK} existe e nao pode ser usado como diretorio." >&2
        exit 1
    fi
fi

mkdir -p "${TARGET_DATASET_DIR}"
chmod 600 "${KAGGLE_CONFIG}"

echo "Baixando dataset em ${TARGET_DATASET_DIR}"
uv run kaggle datasets download \
    -d "${KAGGLE_DATASET}" \
    -p "${TARGET_DATASET_DIR}" \
    --unzip

if [[ "${TARGET_DATASET_DIR}" != "${DATASET_LINK}" ]]; then
    if [[ -L "${DATASET_LINK}" || ! -e "${DATASET_LINK}" ]]; then
        rm -f "${DATASET_LINK}"
    elif [[ -d "${DATASET_LINK}" ]]; then
        if find "${DATASET_LINK}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
            echo "Erro: ${DATASET_LINK} ja existe como diretorio nao vazio." >&2
            echo "Remova ou mova o conteudo antes de recriar o symlink." >&2
            exit 1
        fi
        rmdir "${DATASET_LINK}"
    else
        echo "Erro: ${DATASET_LINK} existe e nao pode ser substituido automaticamente." >&2
        exit 1
    fi

    ln -s "${TARGET_DATASET_DIR}" "${DATASET_LINK}"
    echo "Dataset disponivel em ${DATASET_LINK} (origem: ${TARGET_DATASET_DIR})"
else
    echo "Dataset disponivel em ${DATASET_LINK}"
fi
