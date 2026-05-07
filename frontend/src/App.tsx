import { useState } from "react";
import { Routes, Route, useNavigate, Link } from "react-router-dom";
import { SearchBox } from "./components/SearchBox";
import { Homepage } from "./components/Homepage";
import { SearchResultsRoute } from "./routes/SearchResultsRoute";
import { ArtistRoute } from "./routes/ArtistRoute";
import { SongRoute } from "./routes/SongRoute";

export function App() {
  return (
    <div className="app">
      <Header />
      <main className="main">
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/search" element={<SearchResultsRoute />} />
          <Route path="/artist/:id" element={<ArtistRoute />} />
          <Route path="/song/:id" element={<SongRoute />} />
        </Routes>
      </main>
    </div>
  );
}

function Header() {
  const navigate = useNavigate();
  const [q, setQ] = useState("");
  const handleSearch = (value: string) => {
    if (!value.trim()) return;
    navigate(`/search?q=${encodeURIComponent(value)}`);
  };
  return (
    <header className="header">
      <Link to="/" className="logo">
        músicabr
      </Link>
      <SearchBox value={q} onChange={setQ} onSubmit={handleSearch} compact />
    </header>
  );
}
