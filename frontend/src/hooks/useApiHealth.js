import { useEffect, useState } from "react";
import { getHealth } from "../api/forecas9";

export function useApiHealth(API_BASE) {
  const [apiOnline, setApiOnline] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        await getHealth({ baseUrl: API_BASE });
        if (!cancelled) setApiOnline(true);
      } catch {
        if (!cancelled) setApiOnline(false);
      }
    }

    check();
    const id = setInterval(check, 8000);

    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [API_BASE]);

  return [apiOnline, setApiOnline];
}