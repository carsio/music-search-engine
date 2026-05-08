// Mock dataset — MPB / música brasileira
// Estilo: documentos indexados por um motor de RI

const ARTISTS = {
  "calcinha-preta": {
    id: "calcinha-preta",
    name: "Calcinha Preta",
    tagline: "Banda de forró eletrônico",
    bio: "Banda brasileira de forró eletrônico formada em Aracaju, Sergipe, em 1995. Tornou-se um dos maiores nomes do gênero nos anos 2000, com hits que marcaram o Nordeste e ganharam o país.",
    genres: ["Forró eletrônico", "Forró", "Brega"],
    origin: "Aracaju, SE",
    yearStarted: 1995,
    members: 6,
    monthlyListeners: "2.4M",
    popularity: 78,
    color: "#E0407A",
    albums: [
      { title: "Vol. 8 — Ao Vivo em Recife", year: 2003, tracks: 18 },
      { title: "Vol. 12", year: 2007, tracks: 16 },
      { title: "Calcinha Preta — 20 Anos", year: 2015, tracks: 22 },
      { title: "Acústico", year: 2019, tracks: 14 },
    ],
    topTracks: [
      { title: "Cobertor", album: "Vol. 8", plays: "48M" },
      { title: "Você Não Vale Nada", album: "Vol. 9", plays: "32M" },
      { title: "Quero Ser o DJ", album: "Vol. 12", plays: "21M" },
      { title: "Amor Não é Jogo", album: "Vol. 10", plays: "18M" },
      { title: "Saudade Sem Fim", album: "Vol. 8", plays: "12M" },
    ],
  },
  "joao-gomes": {
    id: "joao-gomes",
    name: "João Gomes",
    tagline: "Cantor de piseiro / forró",
    bio: "Cantor e compositor brasileiro de Serrita, Pernambuco. Estourou nacionalmente em 2021 com 'Meu Pedaço de Pecado' e se tornou um dos principais nomes do piseiro contemporâneo.",
    genres: ["Piseiro", "Forró", "Sertanejo"],
    origin: "Serrita, PE",
    yearStarted: 2020,
    monthlyListeners: "18.2M",
    popularity: 92,
    color: "#2E7D5C",
    albums: [
      { title: "Eu Tenho a Senha", year: 2022, tracks: 12 },
      { title: "Dou-lhe Um — Ao Vivo", year: 2023, tracks: 16 },
      { title: "Raiz", year: 2024, tracks: 14 },
    ],
    topTracks: [
      { title: "Pra Que Fui Me Apaixonar", album: "Raiz", plays: "89M" },
      { title: "Meu Pedaço de Pecado", album: "Eu Tenho a Senha", plays: "412M" },
      { title: "Eu Tenho a Senha", album: "Eu Tenho a Senha", plays: "187M" },
      { title: "Se For Amor", album: "Dou-lhe Um", plays: "98M" },
      { title: "Dengo", album: "Raiz", plays: "64M" },
    ],
  },
};

const SONGS = {
  "cobertor": {
    id: "cobertor",
    title: "Cobertor",
    artistId: "calcinha-preta",
    artist: "Calcinha Preta",
    album: "Vol. 8 — Ao Vivo em Recife",
    year: 2003,
    duration: "4:12",
    plays: "48M",
    composers: ["Daniel Diau"],
    lyrics: `Como é que você foi embora
Sem dizer pelo menos adeus

Como é que você foi embora
Sem dizer pelo menos adeus
E fez sofrer tanto assim
Um coração apaixonado por você

Como é que você depois liga
Pra dizer que está arrependida
Que foi embora
Mas que vai voltar

Pedindo pro meu coração o seu lugar
Ha ha ha
Que pena o meu coração não é mais meu
Mesmo que fosse nunca mais seria seu
Você se foi nem quis saber se estava frio

Eu achei um cobertor
Que me deu tanto amor
E que nunca deixa o frio
Tomar conta de mim

Eu achei um cobertor
Que me deu tanto amor
E que nunca deixa o frio
Tomar conta de mim`,
  },
  "pra-que-fui-me-apaixonar": {
    id: "pra-que-fui-me-apaixonar",
    title: "Pra Que Fui Me Apaixonar",
    artistId: "joao-gomes",
    artist: "João Gomes",
    album: "Raiz",
    year: 2024,
    duration: "3:24",
    plays: "89M",
    composers: ["João Gomes", "Mari Fernandez"],
    lyrics: `E quando o tempo passar, isso não me machucar
Tudo cicatrizar do que levou um fim
Vou guardar o celular, ir pra outro lugar
Onde eu não possa lembrar, eu vou cuidar de mim

Espero que fique bem
Que não conheça ninguém
Que não machuque alguém
Que faz tudo pra lhe ter
Um dia vai aprender
Que o coração não é brinquedo
Pra amar tenho medo

A culpada é você
Eh, eh, eh, eh
Pra que eu fui me apaixonar?
Ah, ah, ah, ah
Tá doendo, mas vai passar
Eh, eh, eh, eh
Pra que eu fui me apaixonar?
Ah, ah, ah, ah
Tá doendo, mas vou superar`,
  },
};

// Web result links — simulam sites externos indexados
const WEB_RESULTS = {
  "calcinha-preta": [
    {
      site: "letras.mus.br",
      url: "letras.mus.br › calcinha-preta",
      title: "Calcinha Preta - Letras, Discografia e Mais",
      snippet: "Conheça as letras das músicas da Calcinha Preta, uma das maiores bandas de forró do Brasil. Mais de 200 letras catalogadas, álbuns ao vivo e clipes oficiais.",
      score: 0.94,
    },
    {
      site: "wikipedia.org",
      url: "pt.wikipedia.org › wiki › Calcinha_Preta",
      title: "Calcinha Preta – Wikipédia, a enciclopédia livre",
      snippet: "Calcinha Preta é uma banda brasileira de forró eletrônico formada em Aracaju, Sergipe, em 1995. A banda alcançou grande sucesso nos anos 2000...",
      score: 0.89,
    },
    {
      site: "vagalume.com.br",
      url: "vagalume.com.br › calcinha-preta",
      title: "Calcinha Preta - Cifras, Letras, Discografia",
      snippet: "Cobertor, Você Não Vale Nada, Quero Ser o DJ e mais. Cifras simplificadas, letras completas e tradução das músicas da banda Calcinha Preta.",
      score: 0.86,
    },
    {
      site: "spotify.com",
      url: "open.spotify.com › artist › calcinha-preta",
      title: "Calcinha Preta | Spotify",
      snippet: "Ouça Calcinha Preta no Spotify. Artista · 2.4M ouvintes mensais. Forró eletrônico direto de Aracaju desde 1995.",
      score: 0.82,
    },
    {
      site: "g1.globo.com",
      url: "g1.globo.com › musica › calcinha-preta-25-anos",
      title: "Calcinha Preta completa 25 anos com show ao vivo",
      snippet: "Banda de forró eletrônico relembra trajetória que começou em Sergipe e ganhou o Brasil com hits como 'Cobertor' e 'Você Não Vale Nada'.",
      score: 0.74,
    },
  ],
  "joao-gomes": [
    {
      site: "letras.mus.br",
      url: "letras.mus.br › joao-gomes",
      title: "João Gomes - Letras das Músicas",
      snippet: "Letras de João Gomes: Pra Que Fui Me Apaixonar, Meu Pedaço de Pecado, Se For Amor e mais. Discografia completa do cantor pernambucano.",
      score: 0.93,
    },
    {
      site: "wikipedia.org",
      url: "pt.wikipedia.org › wiki › João_Gomes_(cantor)",
      title: "João Gomes (cantor) – Wikipédia",
      snippet: "João Gomes Vieira (Serrita, 22 de janeiro de 2002) é um cantor e compositor brasileiro, expoente do piseiro contemporâneo.",
      score: 0.91,
    },
    {
      site: "spotify.com",
      url: "open.spotify.com › artist › joao-gomes",
      title: "João Gomes | Spotify",
      snippet: "18.2M ouvintes mensais. Piseiro e forró direto de Serrita, Pernambuco. Álbuns: Raiz (2024), Dou-lhe Um (2023), Eu Tenho a Senha (2022).",
      score: 0.88,
    },
    {
      site: "rollingstone.com.br",
      url: "rollingstone.com.br › entrevista-joao-gomes",
      title: "João Gomes: o piseiro que conquistou o Brasil",
      snippet: "Em entrevista exclusiva, o cantor de Serrita fala sobre o sucesso de 'Meu Pedaço de Pecado' e os bastidores do álbum Raiz.",
      score: 0.79,
    },
  ],
};

// Resultados de busca por trecho de letra — simulando full-text search
const LYRIC_MATCHES = {
  "coracao": [
    {
      songId: "cobertor",
      title: "Cobertor",
      artist: "Calcinha Preta",
      score: 8.42,
      snippets: [
        { line: 4, text: "Um <mark>coração</mark> apaixonado por você" },
        { line: 13, text: "Pedindo pro meu <mark>coração</mark> o seu lugar" },
        { line: 15, text: "Que pena o meu <mark>coração</mark> não é mais meu" },
      ],
    },
    {
      songId: "pra-que-fui-me-apaixonar",
      title: "Pra Que Fui Me Apaixonar",
      artist: "João Gomes",
      score: 6.18,
      snippets: [
        { line: 11, text: "Que o <mark>coração</mark> não é brinquedo" },
      ],
    },
  ],
  "saudade": [
    {
      songId: "saudade-sem-fim",
      title: "Saudade Sem Fim",
      artist: "Calcinha Preta",
      score: 9.87,
      snippets: [
        { line: 1, text: "<mark>Saudade</mark> sem fim que mora em mim" },
        { line: 7, text: "Essa <mark>saudade</mark> me consome" },
      ],
    },
    {
      songId: "dengo",
      title: "Dengo",
      artist: "João Gomes",
      score: 4.21,
      snippets: [
        { line: 9, text: "Bate uma <mark>saudade</mark> danada" },
      ],
    },
  ],
};

window.MUSIC_DATA = { ARTISTS, SONGS, WEB_RESULTS, LYRIC_MATCHES };
