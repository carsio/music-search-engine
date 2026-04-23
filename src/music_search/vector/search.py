"""Busca semântica na base Spotify vetorizada no Milvus.

Uso como biblioteca:

    from music_search.vector import VectorSearch, search_tracks

    resultados = search_tracks("rock clássico dos anos 70 com guitarra")
    for r in resultados:
        print(r["rank"], f"{r['score']:.4f}", r["track_name"], "—", r["artist_names"])

Uso via CLI (útil para testes rápidos):

    python -m music_search.vector.search "rock clássico anos 70"
    python -m music_search.vector.search "música animada para treinar" --top 10

Configuração por variáveis de ambiente:

    MILVUS_URI      URI do Milvus        (padrão: ./data/vector/milvus_spotify.db)
    USE_OLLAMA      "true" usa Ollama    (padrão: true)
    OLLAMA_URL      endpoint do Ollama   (padrão: http://localhost:11434/v1)
    EMBED_MODEL     modelo Ollama        (padrão: nomic-embed-text)
    OPENAI_API_KEY  se USE_OLLAMA=false, usa OpenAI (text-embedding-3-small)

IMPORTANTE: use o mesmo modelo com que a base foi indexada. Dimensões diferentes
entre indexação e busca geram erro no Milvus.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import TYPE_CHECKING, Any

from music_search.vector.config import COLLECTION_NAME, EmbeddingConfig, default_milvus_uri

if TYPE_CHECKING:
    from openai import OpenAI
    from pymilvus import MilvusClient

_DEFAULT_TOP_K = 20

# Campos recuperados do Milvus (excluindo o vetor).
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


class VectorSearch:
    """Cliente de busca semântica na base Spotify armazenada no Milvus.

    Conexão lazy: o cliente Milvus é criado só na primeira busca. Pode ser usado
    como context manager para garantir `close()` ao encerrar.
    """

    def __init__(
        self,
        milvus_uri: str | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self._milvus_uri = milvus_uri or default_milvus_uri()
        self._embedding_config = embedding_config or EmbeddingConfig.from_env()
        self._embed_client: OpenAI | None = None
        self._milvus: MilvusClient | None = None

    def _ensure_embed_client(self) -> OpenAI:
        if self._embed_client is not None:
            return self._embed_client
        from openai import OpenAI

        cfg = self._embedding_config
        if cfg.use_ollama:
            self._embed_client = OpenAI(base_url=cfg.ollama_url, api_key="ollama")
            log.info("Embedding: Ollama (%s) em %s", cfg.model, cfg.ollama_url)
        else:
            if not cfg.openai_api_key:
                raise RuntimeError(
                    "USE_OLLAMA=false mas OPENAI_API_KEY não está definida. "
                    "Configure uma das duas opções."
                )
            self._embed_client = OpenAI(api_key=cfg.openai_api_key)
            log.info("Embedding: OpenAI (%s)", cfg.model)
        return self._embed_client

    def _ensure_milvus(self) -> MilvusClient:
        if self._milvus is not None:
            return self._milvus
        from pymilvus import MilvusClient

        self._milvus = MilvusClient(uri=self._milvus_uri)
        log.info("Milvus conectado: %s", self._milvus_uri)
        return self._milvus

    def _embed_query(self, texto: str) -> list[float]:
        client = self._ensure_embed_client()
        try:
            resp = client.embeddings.create(model=self._embedding_config.model, input=[texto])
            return resp.data[0].embedding
        except Exception as exc:
            raise RuntimeError(_embed_error_message(exc)) from exc

    def search(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Busca as músicas mais semanticamente próximas do texto.

        Retorna lista ordenada do mais relevante para o menos relevante. Cada
        item contém rank, score (cosseno, [0..1]), metadados da faixa e um
        campo `data_completa` com o registro original deserializado.
        """
        if not query or not query.strip():
            raise ValueError("O texto da busca não pode estar vazio.")

        cliente = self._ensure_milvus()
        vetor_query = self._embed_query(query.strip())

        resultados = cliente.search(
            collection_name=COLLECTION_NAME,
            data=[vetor_query],
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 64}},
        )
        return _format_hits(resultados[0])

    def close(self) -> None:
        """Fecha a conexão com o Milvus. Opcional."""
        if self._milvus is not None:
            self._milvus.close()
            self._milvus = None
            log.info("Conexão Milvus encerrada.")

    def __enter__(self) -> VectorSearch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _embed_error_message(exc: Exception) -> str:
    """Formata a mensagem de erro do embedding.

    Distingue falha de conexão (Ollama/OpenAI offline) de erro da API quando
    o pacote `openai` está disponível. Se não estiver, cai em mensagem genérica.
    """
    try:
        from openai import APIConnectionError

        if isinstance(exc, APIConnectionError):
            return (
                "Não foi possível conectar ao serviço de embedding. "
                "Verifique se o Ollama está rodando (`ollama serve`). "
                f"Detalhe: {exc}"
            )
    except ImportError:
        pass
    return f"Erro na API de embedding: {exc}"


def _format_hits(hits: list) -> list[dict[str, Any]]:
    saida: list[dict[str, Any]] = []
    for posicao, hit in enumerate(hits, start=1):
        entity = hit.get("entity", hit)

        json_raw = entity.get("json_data") or "{}"
        try:
            data_completa = json.loads(json_raw)
        except json.JSONDecodeError:
            data_completa = {}

        saida.append(
            {
                "rank": posicao,
                "score": round(float(hit.get("distance", 0.0)), 6),
                "track_name": entity.get("track_name") or "",
                "artist_names": entity.get("artist_names") or "",
                "album_name": entity.get("album_name") or "",
                "artist_genres": entity.get("artist_genres") or "",
                "album_type": entity.get("album_type") or "",
                "label": entity.get("label") or "",
                "release_date": entity.get("release_date") or "",
                "track_popularity": entity.get("track_popularity", 0),
                "album_popularity": entity.get("album_popularity", 0),
                "duration_ms": entity.get("duration_ms", 0),
                "explicit": bool(entity.get("explicit", False)),
                "data_completa": data_completa,
            }
        )
    return saida


# Singleton por módulo para reaproveitar conexão em chamadas sucessivas.
_instancia_padrao: VectorSearch | None = None


def search_tracks(query: str, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Função de conveniência: busca com a instância padrão do módulo.

    Cria a conexão na primeira chamada e a reutiliza nas seguintes. As
    configurações são lidas das variáveis de ambiente (ver docstring do módulo).
    """
    global _instancia_padrao
    if _instancia_padrao is None:
        _instancia_padrao = VectorSearch()
    return _instancia_padrao.search(query, top_k)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_hit(r: dict[str, Any]) -> None:
    duracao_min = r["duration_ms"] // 60_000
    duracao_seg = (r["duration_ms"] % 60_000) // 1_000
    explicito = "[E] " if r["explicit"] else ""
    ano = r["release_date"][:4] if r["release_date"] else "?"
    print(
        f"  #{r['rank']:02d}  score={r['score']:.4f}  "
        f"{explicito}{r['track_name']}  —  {r['artist_names']}\n"
        f"       Álbum : {r['album_name']}  ({ano})  [{r['album_type']}]\n"
        f"       Gênero: {r['artist_genres'] or '—'}\n"
        f"       Label : {r['label'] or '—'}   "
        f"Pop: {r['track_popularity']}/100   "
        f"Duração: {duracao_min}:{duracao_seg:02d}\n"
    )


def _main_cli() -> None:
    import argparse

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Busca semântica na base Spotify (Milvus + embeddings)",
    )
    parser.add_argument("query", nargs="+", help="Texto de busca")
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=_DEFAULT_TOP_K,
        metavar="N",
        help=f"Número de resultados (padrão: {_DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--milvus",
        default=None,
        metavar="URI",
        help="URI do Milvus (padrão: ./data/vector/milvus_spotify.db)",
    )
    args = parser.parse_args()

    texto = " ".join(args.query)
    print(f'\nBuscando: "{texto}"  (top {args.top})\n{"─" * 60}')

    try:
        with VectorSearch(milvus_uri=args.milvus) as search:
            resultados = search.search(texto, top_k=args.top)
    except RuntimeError as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)

    if not resultados:
        print("Nenhum resultado encontrado.")
        return

    print(f"Encontradas {len(resultados)} músicas:\n")
    for r in resultados:
        _print_hit(r)


if __name__ == "__main__":
    _main_cli()
