export type LiveState = "connecting" | "live" | "reconnecting" | "polling" | "paused";

export interface LiveCallbacks {
  synchronize: (options: { status: boolean; panels: boolean }) => Promise<void>;
  training: (payload: Record<string, any>) => void;
  state: (state: LiveState) => void;
}

// One refresh at a time, with EventSource's reconnect and a polling fallback.
// Keep this independent of Vue so lifecycle and network recovery can be tested.
type LiveHost = Pick<Window, "setTimeout" | "clearTimeout" | "addEventListener" | "removeEventListener"> & { EventSource?: typeof EventSource };

export function createLiveUpdates(callbacks: LiveCallbacks, host: LiveHost = window, page: Document = document) {
  let source: EventSource | null = null;
  let timer = 0;
  let stopped = true;
  let connected = false;
  let running = false;
  let pending = false;
  let lastPanels = 0;
  let revision = "";
  let lastEvent = 0;

  function schedule(): void {
    host.clearTimeout(timer);
    if (!stopped && page.visibilityState !== "hidden") timer = host.setTimeout(() => void tick(), 1500);
  }

  async function tick(): Promise<void> {
    if (stopped || page.visibilityState === "hidden" || running) return;
    host.clearTimeout(timer);
    if (connected && Date.now() - lastEvent > 20000) {
      close();
      callbacks.state("reconnecting");
    }
    connect();
    const panels = pending || Date.now() - lastPanels >= 5000;
    pending = false;
    if (connected && !panels) { schedule(); return; }
    running = true;
    try {
      await callbacks.synchronize({ status: !connected, panels });
      if (panels) lastPanels = Date.now();
    } catch {
      // A failed request must not disable future refreshes.
    } finally {
      running = false;
      if (pending && !stopped) void tick();
      else schedule();
    }
  }

  function refresh(): void {
    pending = true;
    void tick();
  }

  function close(): void {
    source?.close();
    source = null;
    connected = false;
    host.clearTimeout(timer);
  }

  function connect(): void {
    if (source || stopped || page.visibilityState === "hidden") return;
    if (typeof host.EventSource !== "function") { callbacks.state("polling"); return; }
    callbacks.state("connecting");
    let stream: EventSource;
    try { stream = new host.EventSource("/api/events"); }
    catch { callbacks.state("polling"); return; }
    source = stream;
    stream.onopen = () => {
      if (source !== stream) return;
      connected = true;
      lastEvent = Date.now();
      callbacks.state("live");
      refresh();
    };
    stream.addEventListener("trainer", (event) => {
      if (source !== stream || stopped) return;
      try {
        const payload = JSON.parse((event as MessageEvent).data);
        if (!payload.training || typeof payload.revision !== "string") return;
        lastEvent = Date.now();
        callbacks.training(payload.training);
        if (revision !== payload.revision) { revision = payload.revision; refresh(); }
      } catch {
        // Ignore a malformed event; the next snapshot and polling recover it.
      }
    });
    stream.addEventListener("heartbeat", () => { if (source === stream) lastEvent = Date.now(); });
    stream.onerror = () => {
      if (source !== stream) return;
      connected = false;
      callbacks.state("reconnecting");
      if (stream.readyState === 2) { stream.close(); source = null; }
      refresh();
    };
  }

  function resume(): void {
    if (stopped) return;
    if (page.visibilityState === "hidden") {
      close();
      callbacks.state("paused");
      return;
    }
    connect();
    refresh();
  }

  return {
    refresh,
    start(): void {
      if (!stopped) return;
      stopped = false;
      page.addEventListener("visibilitychange", resume);
      host.addEventListener("focus", resume);
      host.addEventListener("online", resume);
      resume();
    },
    stop(): void {
      stopped = true;
      pending = false;
      close();
      page.removeEventListener("visibilitychange", resume);
      host.removeEventListener("focus", resume);
      host.removeEventListener("online", resume);
    },
  };
}
