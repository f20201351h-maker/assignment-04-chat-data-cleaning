// Build the static site into site/dist and refuse to build if something is off.
//   node site/build.mjs
// Checks: data file present and parseable, every data-k binding in index.html
// resolves to a value in data/site.json, no secrets or raw emails in the bundle.
import { readFileSync, writeFileSync, mkdirSync, rmSync, readdirSync, statSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
const files = ["index.html", "style.css", "app.js", "favicon.svg", "data/site.json"];

const fail = (msg) => { console.error("BUILD FAILED: " + msg); process.exit(1); };

const data = JSON.parse(readFileSync(join(here, "data/site.json"), "utf8"));
const html = readFileSync(join(here, "index.html"), "utf8");

// every bound key must exist in the generated data
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
const keys = [...html.matchAll(/data-k="([^"]+)"/g)].map((m) => m[1]);
const missing = keys.filter((k) => get(data, k) === undefined);
if (missing.length) fail("unbound data keys: " + missing.join(", "));

// nothing secret or personal goes out
const bundle = files.map((f) => readFileSync(join(here, f), "utf8")).join("\n");
const secretRe = /(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|EXA_API_KEY|HF_TOKEN|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/;
const s = bundle.match(secretRe);
if (s && !/canary/i.test(bundle.slice(Math.max(0, s.index - 80), s.index))) fail("secret-like string in bundle: " + s[0].slice(0, 8) + "...");
const emailRe = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;
const emails = [...bundle.matchAll(emailRe)].map((m) => m[0])
  .filter((e) => !/@(?:[\w.-]+\.)?(example\.(com|net|org)|[\w-]+\.(test|example|invalid))$/i.test(e));
if (emails.length) fail("email address in bundle: " + emails.length);

rmSync(dist, { recursive: true, force: true });
for (const f of files) {
  mkdirSync(dirname(join(dist, f)), { recursive: true });
  copyFileSync(join(here, f), join(dist, f));
}
const size = files.reduce((n, f) => n + statSync(join(dist, f)).size, 0);
console.log(`built ${files.length} files, ${(size / 1024).toFixed(1)} KB, ${keys.length} data bindings checked`);
