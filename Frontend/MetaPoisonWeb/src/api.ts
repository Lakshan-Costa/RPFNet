
const LOCAL_API = import.meta.env.VITE_API_BASE;

async function fetchWithFallback(url: string, options?: RequestInit) {
  try {
    const res = await fetch(`/api${url}`, options);

    if (!res.ok) throw new Error("Proxy failed");

    return res;
  } catch (err) {
    console.warn("Proxy failed → switching to LOCAL");

    // fallback to local only in dev
    return fetch(`${LOCAL_API}${url}`, options);
  }
}

export default fetchWithFallback;