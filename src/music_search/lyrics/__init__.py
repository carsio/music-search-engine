"""Pipeline de extracao de letras com cache persistente e cascata de fontes."""

from music_search.lyrics.cache import LyricsCache

__all__ = ["LyricsCache", "PipelineConfig", "run_pipeline"]


def __getattr__(name: str):
    if name in {"PipelineConfig", "run_pipeline"}:
        from music_search.lyrics.pipeline import PipelineConfig, run_pipeline

        return {"PipelineConfig": PipelineConfig, "run_pipeline": run_pipeline}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
