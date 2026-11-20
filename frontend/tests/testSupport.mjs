export function harness() {
  let now = 100000;
  let nextTimer = 0;
  const timers = new Map();
  const page = new EventTarget();
  page.visibilityState = "visible";
  const host = new EventTarget();
  host.setTimeout = (callback, delay) => { const id = ++nextTimer; timers.set(id, { at: now + delay, callback }); return id; };
  host.clearTimeout = id => timers.delete(id);
  const streams = [];
  host.EventSource = class extends EventTarget {
    readyState = 0;
    closed = false;
    constructor(url) { super(); this.url = url; streams.push(this); }
    open() { this.readyState = 1; this.onopen?.(); }
    fail(closed = false) { this.readyState = closed ? 2 : 0; this.onerror?.(); }
    close() { this.closed = true; this.readyState = 2; }
    event(name, payload) { this.dispatchEvent(new MessageEvent(name, { data: JSON.stringify(payload) })); }
  };
  return {
    host, page, streams, timers,
    now: () => now,
    async advance(ms) {
      const end = now + ms;
      for (;;) {
        const pending = [...timers].filter(([, value]) => value.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!pending) break;
        now = pending[1].at; timers.delete(pending[0]); pending[1].callback(); await flush();
      }
      now = end; await flush();
    },
  };
}
export async function flush() { for (let n = 0; n < 8; n++) await new Promise(resolve => setImmediate(resolve)); }
