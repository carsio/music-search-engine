"""
Módulo de busca semântica na base vetorial Spotify (Milvus).

Uso como módulo (import em outra aplicação):
    from spotify_search import buscar_musicas

    resultados = buscar_musicas("rock clássico dos anos 70 com guitarra")
    for r in resultados:
        print(r["rank"], f'{r["score"]:.4f}', r["track_name"], "—", r["artist_names"])

Uso direto na linha de comando (para testes):
    python spotify_search.py "rock clássico anos 70"
    python spotify_search.py "música eletrônica animada para treinar" --top 10

Configuração via variáveis de ambiente (ou parâmetros diretos no construtor):
    MILVUS_URI      URL do Milvus                        (padrão: http://localhost:19530)
    OLLAMA_URL      endpoint do Ollama                   (padrão: http://localhost:11434/v1)
    EMBED_MODEL     modelo de embedding do Ollama        (padrão: nomic-embed-text)
    USE_OLLAMA      'true' para forçar Ollama            (padrão: true)
    OPENAI_API_KEY  se USE_OLLAMA != 'true', usa OpenAI em vez do Ollama

IMPORTANTE: use o mesmo modelo com que a base foi indexada.
    Base gerada com nomic-embed-text (Ollama, 768 dims) → mantenha USE_OLLAMA=true.
    Base gerada com text-embedding-3-small (OpenAI, 1536 dims) → defina OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI, APIConnectionError, APIError
from pymilvus import MilvusClient

# ── Constantes ────────────────────────────────────────────────────────────────

_BASE_DIR       = Path(__file__).parent
_COLLECTION     = "spotify_tracks"
_DEFAULT_TOP_K  = 20

# Campos recuperados do Milvus (excluindo o vetor, que não precisa retornar)
_OUTPUT_FIELDS = [
    "id",
    "track_name",
    "artist_names",
    "album_name",
    "artist_genres",
    "album_type",
    "label",
    "release_date",
    "track_popularity",
    "album_popularity",
    "duration_ms",
    "explicit",
    "json_data",
]

log = logging.getLogger(__name__)


# ── Classe principal ──────────────────────────────────────────────────────────

class SpotifySearch:
    """
    Cliente de busca semântica na base Spotify armazenada no Milvus.

    Pode ser instanciado diretamente com parâmetros ou deixar os valores
    padrão serem lidos das variáveis de ambiente.

    Parâmetros
    ----------
    milvus_uri : str, opcional
        Caminho do arquivo Milvus Lite (.db) ou URL de instância remota.
        Padrão: variável MILVUS_URI ou './milvus_spotify.db'
    ollama_url : str, opcional
        URL base da API OpenAI-compatível do Ollama.
        Padrão: variável OLLAMA_URL ou 'http://localhost:11434/v1'
    embed_model : str, opcional
        Nome do modelo de embedding rodando no Ollama.
        Padrão: variável EMBED_MODEL ou 'nomic-embed-text'
    openai_api_key : str, opcional
        Se fornecida, usa a OpenAI API em vez do Ollama
        (modelo text-embedding-3-small, 1536 dims).
        Padrão: variável OPENAI_API_KEY (se ausente, usa Ollama)
    """

    def __init__(
        self,
        milvus_uri: str | None = None,
        ollama_url: str | None = None,
        embed_model: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        self._milvus_uri = (
            milvus_uri
            or os.getenv("MILVUS_URI", "http://localhost:19530")
        )

        use_ollama = os.getenv("USE_OLLAMA", "true").lower() != "false"
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if api_key and not use_ollama:
            self._embed_client = OpenAI(api_key=api_key)
            self._embed_model  = "text-embedding-3-small"
            log.info("Embedding: OpenAI API (text-embedding-3-small)")
        else:
            url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
            self._embed_model = embed_model or os.getenv("EMBED_MODEL", "nomic-embed-text")
            self._embed_client = OpenAI(base_url=url, api_key="ollama")
            log.info(f"Embedding: Ollama ({self._embed_model}) em {url}")

        self._milvus: MilvusClient | None = None

    # ── Conexão lazy ──────────────────────────────────────────────────────────

    def _conectar(self) -> MilvusClient:
        if self._milvus is None:
            self._milvus = MilvusClient(uri=self._milvus_uri)
            log.info(f"Milvus conectado: {self._milvus_uri}")
        return self._milvus

    # ── Embedding da consulta ─────────────────────────────────────────────────

    def _embed_query(self, texto: str) -> list[float]:
        try:
            resp = self._embed_client.embeddings.create(
                model=self._embed_model,
                input=[texto],
            )
            return resp.data[0].embedding
        except APIConnectionError as exc:
            raise RuntimeError(
                f"Não foi possível conectar ao serviço de embedding. "
                f"Verifique se o Ollama está rodando ('ollama serve'). Detalhe: {exc}"
            ) from exc
        except APIError as exc:
            raise RuntimeError(f"Erro na API de embedding: {exc}") from exc

    # ── Busca principal ───────────────────────────────────────────────────────

    def buscar(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Busca as músicas mais semanticamente próximas do texto fornecido.

        Parâmetros
        ----------
        query : str
            Texto descrevendo o que se quer encontrar.
            Exemplos: "rock clássico anos 70", "música animada para academia",
                      "jazz melancólico de piano", "trap brasileiro"
        top_k : int
            Número máximo de resultados (padrão: 20).

        Retorna
        -------
        list[dict]
            Lista ordenada da mais relevante para a menos relevante.
            Cada item contém:
                rank           : posição na lista (1 = melhor correspondência)
                score          : similaridade cosseno [0..1], maior = mais relevante
                track_name     : nome da faixa
                artist_names   : artistas separados por ' | '
                album_name     : nome do álbum
                artist_genres  : gêneros separados por ' | '
                album_type     : single / album / compilation
                label          : gravadora
                release_date   : data de lançamento
                track_popularity : popularidade da faixa (0-100)
                album_popularity : popularidade do álbum (0-100)
                duration_ms    : duração em milissegundos
                explicit       : conteúdo explícito (bool)
                data_completa  : dict com todos os campos originais do CSV
        """
        if not query or not query.strip():
            raise ValueError("O texto da busca não pode estar vazio.")

        cliente = self._conectar()

        vetor_query = self._embed_query(query.strip())

        resultados_brutos = cliente.search(
            collection_name=_COLLECTION,
            data=[vetor_query],
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        )

        return self._formatar_resultados(resultados_brutos[0])

    # ── Formatação da resposta ────────────────────────────────────────────────

    @staticmethod
    def _formatar_resultados(hits: list) -> list[dict[str, Any]]:
        saida = []
        for posicao, hit in enumerate(hits, start=1):
            entity = hit.get("entity", hit)  # compatibilidade MilvusClient

            # Deserializa o JSON completo armazenado
            json_raw = entity.get("json_data") or "{}"
            try:
                data_completa = json.loads(json_raw)
            except json.JSONDecodeError:
                data_completa = {}

            saida.append({
                "rank":             posicao,
                "score":            round(float(hit.get("distance", 0.0)), 6),
                "track_name":       entity.get("track_name") or "",
                "artist_names":     entity.get("artist_names") or "",
                "album_name":       entity.get("album_name") or "",
                "artist_genres":    entity.get("artist_genres") or "",
                "album_type":       entity.get("album_type") or "",
                "label":            entity.get("label") or "",
                "release_date":     entity.get("release_date") or "",
                "track_popularity": entity.get("track_popularity", 0),
                "album_popularity": entity.get("album_popularity", 0),
                "duration_ms":      entity.get("duration_ms", 0),
                "explicit":         bool(entity.get("explicit", False)),
                "data_completa":    data_completa,
            })

        return saida

    # ── Utilitários ───────────────────────────────────────────────────────────

    def fechar(self) -> None:
        """Fecha a conexão com o Milvus. Opcional — use ao encerrar a aplicação."""
        if self._milvus is not None:
            self._milvus.close()
            self._milvus = None
            log.info("Conexão Milvus encerrada.")

    def __enter__(self) -> "SpotifySearch":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()


# ── Instância padrão (singleton por módulo) ───────────────────────────────────

_instancia_padrao: SpotifySearch | None = None


def buscar_musicas(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Função de conveniência: busca músicas usando a instância padrão do módulo.

    Cria a conexão na primeira chamada e a reutiliza nas subsequentes.
    As configurações são lidas das variáveis de ambiente (ver docstring do módulo).

    Parâmetros
    ----------
    query : str
        Texto livre descrevendo a música desejada.
    top_k : int
        Número máximo de resultados (padrão: 20).

    Retorna
    -------
    list[dict]  — ver SpotifySearch.buscar() para estrutura completa.
    """
    global _instancia_padrao
    if _instancia_padrao is None:
        _instancia_padrao = SpotifySearch()
    return _instancia_padrao.buscar(query, top_k)


# ── CLI para testes rápidos ───────────────────────────────────────────────────

def _exibir_resultado(r: dict) -> None:
    duracao_min = r["duration_ms"] // 60_000
    duracao_seg = (r["duration_ms"] % 60_000) // 1_000
    explicito   = "[E] " if r["explicit"] else ""
    print(
        f"  #{r['rank']:02d}  score={r['score']:.4f}  "
        f"{explicito}{r['track_name']}  —  {r['artist_names']}\n"
        f"       Album : {r['album_name']}  ({r['release_date'][:4] if r['release_date'] else '?'})"
        f"  [{r['album_type']}]\n"
        f"       Gênero: {r['artist_genres'] or '—'}\n"
        f"       Label : {r['label'] or '—'}   "
        f"Pop: {r['track_popularity']}/100   "
        f"Duração: {duracao_min}:{duracao_seg:02d}\n"
    )


def _main_cli() -> None:
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Busca semântica na base Spotify (Milvus + Ollama)"
    )
    parser.add_argument("query", nargs="+", help="Texto de busca")
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=_DEFAULT_TOP_K,
        metavar="N",
        help=f"Número de resultados (padrão: {_DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--milvus",
        default=None,
        metavar="URI",
        help="URI do Milvus (padrão: ./milvus_spotify.db)",
    )
    args = parser.parse_args()

    texto = " ".join(args.query)
    print(f'\nBuscando: "{texto}"  (top {args.top})\n{"─" * 60}')

    try:
        with SpotifySearch(milvus_uri=args.milvus) as search:
            resultados = search.buscar(texto, top_k=args.top)
    except RuntimeError as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)

    if not resultados:
        print("Nenhum resultado encontrado.")
        return

    print(f"Encontradas {len(resultados)} músicas:\n")
    for r in resultados:
        _exibir_resultado(r)


if __name__ == "__main__":
    _main_cli()
