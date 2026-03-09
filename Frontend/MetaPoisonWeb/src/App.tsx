import { Routes, Route } from "react-router-dom";
import Dashboard from "./Dashboard";

export default function App() {
  return (
    <div role="application" aria-label="Data Poisoning Detection System">
      <Routes>
        <Route path="/" element={<Dashboard />} />
      </Routes>
    </div>
  );
}
