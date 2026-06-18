# Relatório técnico — Recuperação da Informação

Artigo (template SBC) do trabalho da disciplina **Recuperação da Informação**
(UFAM 2026/1), descrevendo o *Music Search Engine* — desenvolvimento e avaliação
experimental de um sistema de RI multi-campo e híbrido para músicas brasileiras.

## Arquivos

| Arquivo | Função |
| --- | --- |
| `main.tex` | o artigo (LaTeX) |
| `sbc-template.sty` | folha de estilo SBC vendorizada (auto-contida, compila offline) |
| `refs.bib` | referências bibliográficas |
| `figures/` | figuras dos resultados (heatmap, barras por intent, boxplot, scatter) |
| `benchmark.py` | script auxiliar de *benchmark* de **performance** (latência/vocabulário) — não é mais citado no relatório |
| `results.md` | saída do `benchmark.py` (apenas performance) |

Os números da seção de Resultados vêm da **coleção de referência** estilo TREC
(50 consultas, 1.604 julgamentos *silver standard*) descrita no próprio artigo —
o pipeline de avaliação de relevância é externo a este diretório.

## Compilar localmente

Requer [Tectonic](https://tectonic-typesetting.github.io/) (engine LaTeX
auto-contido, baixa os pacotes do CTAN sob demanda):

```bash
# macOS
brew install tectonic

# Compilar (gera main.pdf)
cd docs/report
tectonic main.tex
```

O PDF (`main.pdf`) não é versionado — o artefato canônico é produzido pela
pipeline `.github/workflows/report.yml`. Em cada push para `master`, a pipeline
publica o PDF num **release de tag móvel** `report-latest`, de modo que o link no
README do projeto aponta sempre para a última versão gerada:

<https://github.com/carsio/music-search-engine/releases/download/report-latest/main.pdf>

Em PRs e outras branches o PDF fica disponível como *artifact* da execução.

## Regenerar os números do benchmark

```bash
# da raiz do repositório
uv run python docs/report/benchmark.py
```

Isso reconstrói o índice sobre `data/derived/final/br_curated_lyrics.parquet`,
mede tempo de indexação, vocabulário e latência (BM25 e TF-IDF), e atualiza
`results.md`. Os valores devem então ser transcritos para `main.tex`.
