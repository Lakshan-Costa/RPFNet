

const AWS_API = import.meta.env.VITE_API_AWS;
const LOCAL_API = import.meta.env.VITE_API_BASE;

async function fetchWithFallback(url: string, options?: RequestInit) {
  try {
    // Try AWS first
    const res = await fetch(`${AWS_API}${url}`, options);

    if (!res.ok) throw new Error("AWS failed");

    return res;
  } catch (err) {
    console.warn("AWS failed → switching to LOCAL");

    // Fallback to local
    const res = await fetch(`${LOCAL_API}${url}`, options);
    return res;
  }
}

export default fetchWithFallback;