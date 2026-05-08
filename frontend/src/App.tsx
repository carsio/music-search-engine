import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { ArtistDetailPage } from "./views/ArtistDetailPage";
import { SongDetailPage } from "./views/SongDetailPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />} />
      <Route path="/artist/:id" element={<ArtistDetailPage />} />
      <Route path="/song/:id" element={<SongDetailPage />} />
      <Route path="/search" element={<Navigate to="/" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
