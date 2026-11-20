import assert from "node:assert/strict";
import { test } from "node:test";
import { createLiveUpdates } from "../src/liveUpdates.ts";

import { harness, flush } from "./testSupport.mjs";

function setup(t, synchronize = async () => {}) {
  const env = harness();
  t.mock.method(Date, "now", env.now);
  const states = [], training = [], calls = [];
  const live = createLiveUpdates({
    synchronize: async options => { calls.push(options); await synchronize(options); },
    training: payload => training.push(payload), state: state => states.push(state),
  }, env.host, env.page);
  live.start();
  t.after(() => live.stop());
  return { ...env, live, states, training, calls };
}

test("streams training and coalesces invalidation while a refresh is in flight", async t => {
  let release;
  let concurrent = 0, maximum = 0;
  const env = setup(t, async () => { maximum = Math.max(maximum, ++concurrent); await new Promise(resolve => { release = resolve; }); concurrent--; });
  const source = env.streams[0];
  source.open();
  source.event("trainer", { training: { running: true }, revision: "first:1" });
  source.event("trainer", { training: { running: true, progress: 20 }, revision: "first:2" });
  assert.equal(env.training.length, 2);
  assert.equal(env.calls.length, 1);
  release(); await flush();
  assert.equal(env.calls.length, 2);
  assert.equal(maximum, 1);
  release(); await flush();
  const before = env.calls.length;
  source.event("trainer", { training: { running: true, progress: 30 }, revision: "first:2" });
  await flush();
  assert.equal(env.calls.length, before, "log changes alone do not rescan the library");
});

test("fallback polls after a disconnect and resyncs after a server restart", async t => {
  const env = setup(t); await flush();
  const source = env.streams[0]; source.open(); await flush();
  source.fail(); await flush();
  assert.equal(env.states.at(-1), "reconnecting");
  const count = env.calls.length;
  await env.advance(1600);
  assert.ok(env.calls.length > count);
  assert.equal(env.calls.at(-1).status, true);
  source.open(); source.event("trainer", { training: { running: false }, revision: "restarted:0" }); await flush();
  assert.equal(env.states.at(-1), "live");
  assert.equal(env.calls.at(-1).panels, true);
  assert.equal(env.calls.at(-1).status, false);
});

test("pauses in hidden tabs, resumes on focus/online, and cleans up", async t => {
  const env = setup(t); await flush();
  env.page.visibilityState = "hidden";
  env.page.dispatchEvent(new Event("visibilitychange"));
  assert.equal(env.streams[0].closed, true);
  assert.equal(env.timers.size, 0);
  const count = env.calls.length; await env.advance(30000);
  assert.equal(env.calls.length, count);
  env.page.visibilityState = "visible";
  env.page.dispatchEvent(new Event("visibilitychange")); await flush();
  assert.equal(env.streams.length, 2);
  env.host.dispatchEvent(new Event("online")); await flush();
  assert.ok(env.calls.length > count);
  env.live.stop();
  env.host.dispatchEvent(new Event("focus")); env.host.dispatchEvent(new Event("online")); await env.advance(30000);
  assert.equal(env.timers.size, 0);
  assert.equal(env.streams.length, 2);
});

test("recovers from rejected refreshes, malformed events, and stalled streams", async t => {
  const env = setup(t, async () => { throw Error("offline"); }); await flush();
  const source = env.streams[0]; source.open(); await flush();
  source.dispatchEvent(new MessageEvent("trainer", { data: "broken" }));
  await env.advance(22000);
  assert.equal(source.closed, true);
  assert.equal(env.streams.length, 2);
  assert.ok(env.calls.length > 2);
});

test("heartbeats keep an idle stream alive and missing EventSource uses polling", async t => {
  const env = setup(t); await flush();
  env.streams[0].open(); await flush();
  for (let i = 0; i < 3; i++) { await env.advance(10000); env.streams[0].event("heartbeat", {}); }
  assert.equal(env.streams.length, 1);
  env.live.stop();
  delete env.host.EventSource;
  const states = [], calls = [];
  const fallback = createLiveUpdates({ synchronize: async o => { calls.push(o); }, state: s => states.push(s), training: () => {} }, env.host, env.page);
  fallback.start(); await flush(); await env.advance(1600); fallback.stop();
  assert.equal(states.at(-1), "polling");
  assert.ok(calls.length >= 2);
  assert.ok(calls.every(call => call.status));
});
