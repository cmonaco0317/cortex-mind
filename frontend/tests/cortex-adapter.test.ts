import { describe, it, expect } from "vitest";
import {
  AGENT_TRACE_ADAPTER_VERSION,
  parseJsonlTrace,
  parseJsonArrayRecords,
  entryText,
} from "../src/cortex/adapters/agent-trace";

// This adapter parses third-party schemas Cortex doesn't own, so it is
// explicitly versioned and must fail SOFT (never throw) — these tests pin that
// contract so a future upstream format change is caught here, in isolation.
describe("agent-trace adapter (versioned, may-break)", () => {
  it("exposes a semver version so breakage is traceable", () => {
    expect(AGENT_TRACE_ADAPTER_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it("parses Claude Code content-block assistant turns", () => {
    const jsonl = JSON.stringify({
      type: "assistant",
      message: { role: "assistant", content: [{ type: "text", text: "Investigating the failing token refresh in the auth middleware." }] },
    });
    const r = parseJsonlTrace(jsonl, "session.jsonl");
    expect(r.concepts).toHaveLength(1);
    expect(r.concepts[0].text).toContain("token refresh");
    expect(r.concepts[0].domain).toBe("assistant");
  });

  it("fails soft (no throw) on garbage and says why", () => {
    const r = parseJsonlTrace("garbage\nmore garbage", "x.jsonl");
    expect(r.concepts).toHaveLength(0);
    expect(r.notes[0].level).toBe("error");
  });

  it("rejects a JSON object where an array is expected, with guidance", () => {
    const r = parseJsonArrayRecords(JSON.stringify({ foo: "bar" }), "d.json");
    expect(r.concepts).toHaveLength(0);
    expect(r.notes[0].message).toMatch(/array/i);
  });

  it("extracts text across the known trace shapes", () => {
    expect(entryText({ text: "plain text field" })).toContain("plain text");
    expect(entryText({ message: { content: "message content string" } })).toContain("message content");
    expect(entryText({ content: [{ type: "text", text: "block text" }] })).toContain("block text");
  });
});
