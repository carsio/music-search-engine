# Relatório técnico — ICC222

Artigo (template SBC) do trabalho da disciplina **ICC222 — Tópicos em
Recuperação de Informação** (UFAM 2026/1), descrevendo o *Music Search Engine*.

## Arquivos

| Arquivo | Função |
| --- | --- |
| `main.tex` | o artigo (LaTeX, ~10 páginas) |
| `sbc-template.sty` | folha de estilo SBC vendorizada (auto-contida, compila offline) |
| `refs.bib` | referências bibliográficas |
| `benchmark.py` | script reproduzível que gera os números da seção de Resultados |
| `results.md` | saída do benchmark (rastreabilidade dos números citados no `.tex`) |

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
pipeline `.github/workflows/report.yml` e fica disponível como *artifact* em
cada execução do GitHub Actions.

## Regenerar os números do benchmark

```bash
# da raiz do repositório
uv run python docs/report/benchmark.py
```

Isso reconstrói o índice sobre `data/derived/final/br_curated_lyrics.parquet`,
mede tempo de indexação, vocabulário e latência (BM25 e TF-IDF), e atualiza
`results.md`. Os valores devem então ser transcritos para `main.tex`.
