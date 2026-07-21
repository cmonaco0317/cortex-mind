// Stable, dependency-free text primitives shared by the ingest core and the
// (may-break) agent-trace adapter. Kept in its own module so the adapter never
// has to import back from ingest.ts (no import cycle) and so the security-
// sensitive redaction lives in one place.

export interface Concept {
  id: string;
  label: string;
  domain: string;
  text: string;
}

/** A human-readable diagnostic surfaced to the user ("couldn't parse, here's why"). */
export interface ParseNote {
  file: string;
  level: "info" | "warn" | "error";
  message: string;
}

export function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "c";
}

export function firstWords(s: string, n: number): string {
  return s.split(/\s+/).slice(0, n).join(" ");
}

// Scrub common secrets from ingested text BEFORE it becomes a neuron/insight/
// shareable card. Agent traces (.jsonl) routinely carry API keys, tokens, and
// private keys; without this, the watermarked share card is a leak vector.
export function redactSecrets(s: string): string {
  return s
    .replace(/-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----/g, "[REDACTED KEY]")
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}/g, "[REDACTED JWT]")
    .replace(/\b(?:sk|pk|rk)[-_](?:live|test|proj|ant)?[-_]?[A-Za-z0-9]{16,}\b/gi, "[REDACTED]")
    .replace(/\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g, "[REDACTED]")
    .replace(/\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g, "[REDACTED]")
    .replace(/\bAKIA[0-9A-Z]{16}\b/g, "[REDACTED]")
    .replace(/\bAIza[0-9A-Za-z_-]{20,}\b/g, "[REDACTED]")
    .replace(/\bBearer\s+[A-Za-z0-9._-]{16,}/gi, "Bearer [REDACTED]")
    .replace(/\b(api[_-]?key|secret|token|password|passwd|authorization)\b(["']?\s*[:=]\s*["']?)[A-Za-z0-9._-]{12,}/gi, "$1$2[REDACTED]");
}

export function redactConcept(c: Concept): Concept {
  return { ...c, label: redactSecrets(c.label), text: redactSecrets(c.text) };
}
