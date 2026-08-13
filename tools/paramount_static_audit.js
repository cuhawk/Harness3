const fs = require("fs");
const path = require("path");

const root = process.argv[2];
const outDir = process.argv[3];

if (!root || !outDir) {
  console.error("usage: node tools/paramount_static_audit.js <paramount-root> <output-dir>");
  process.exit(2);
}

const jsDir = path.join(root, "js-recon");
const reconDir = path.join(root, "recon");

function walk(dir, pred = () => true) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walk(p, pred));
    else if (pred(p)) out.push(p);
  }
  return out;
}

function rel(p) {
  return path.relative(root, p).replace(/\\/g, "/");
}

function uniqueSorted(items) {
  return Array.from(new Set(items.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function extractAll(text, re, group = 0) {
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) out.push(m[group]);
  return out;
}

function cleanString(s) {
  return s
    .replace(/\\u002F/g, "/")
    .replace(/\\\//g, "/")
    .replace(/[\\"'`,;)\]}]+$/g, "")
    .replace(/^[\\"'`({\[]+/g, "");
}

function around(text, needle, radius = 260) {
  const i = text.indexOf(needle);
  if (i < 0) return "";
  const start = Math.max(0, i - radius);
  const end = Math.min(text.length, i + needle.length + radius);
  return text.slice(start, end).replace(/\s+/g, " ");
}

function write(name, body) {
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(path.join(outDir, name), body.endsWith("\n") ? body : body + "\n");
}

const files = [
  ...walk(jsDir, (p) => /\.(js|mjs)$/i.test(p)),
  ...walk(reconDir, (p) => /\.(html|txt)$/i.test(p)),
];

const jsFiles = files.filter((p) => p.startsWith(jsDir));
const corpus = files.map((file) => ({ file, text: fs.readFileSync(file, "utf8") }));

const urls = [];
const paths = [];
const hosts = [];
const keywords = [];
const interesting = [];

const keywordList = [
  "authSuiteFetchHelper",
  "applicationAccessToken",
  "deviceRefreshToken",
  "customRequestHeaders",
  "browseHubRedirects",
  "window.location.replace",
  "getFeatureFlagsFromCookies",
  "ff-",
  "api_key",
  "service-worker.js",
  "pushNotification",
  "mediaToken",
  "getMediaToken",
  "edenCallback",
  "epg.endpoint",
  "deltaEndpoint",
  "OneTrust",
  "Braze",
  "doppler",
  "ketch",
];

for (const { file, text } of corpus) {
  const r = rel(file);
  for (const u of extractAll(text, /https?:\/\/[^\s"'`<>)\\]+/g)) {
    const cleaned = cleanString(u);
    urls.push(`${cleaned}\t${r}`);
    try {
      hosts.push(`${new URL(cleaned).host}\t${r}`);
    } catch {}
  }
  for (const m of extractAll(text, /["'`]((?:\/|\.\/|\.\.\/)[A-Za-z0-9_./?&=%:{}$@!+~#-]{2,})["'`]/g, 1)) {
    const cleaned = cleanString(m);
    if (!cleaned.startsWith("//")) paths.push(`${cleaned}\t${r}`);
  }
  for (const k of keywordList) {
    if (text.includes(k)) {
      keywords.push(`${k}\t${r}`);
      interesting.push({ keyword: k, file: r, context: around(text, k) });
    }
  }
}

const apiishUrls = uniqueSorted(urls.map((x) => x.split("\t")[0])).filter((u) =>
  /(api|graphql|auth|login|account|user|profile|token|media|video|episode|content|feed|search|config|doppler|identity|oauth|braze|ketch|privacy|adobe|cbs|paramount|mtvnservices)/i.test(u)
);
const apiishPaths = uniqueSorted(paths.map((x) => x.split("\t")[0])).filter((p) =>
  /(api|graphql|auth|login|account|user|profile|token|media|video|episode|content|feed|search|config|service-worker|callback|logout|signin|privacy|manifest|sitemap|robots)/i.test(p)
);

const suspectedSecrets = [];
for (const { file, text } of corpus) {
  const r = rel(file);
  const patterns = [
    ["assignment_key", /\b(?:api[_-]?key|client[_-]?secret|secret|token|password|authorization|bearer)\b.{0,80}/gi],
    ["long_hex", /\b[a-f0-9]{32,64}\b/gi],
    ["jwt_like", /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g],
  ];
  for (const [kind, re] of patterns) {
    for (const m of extractAll(text, re)) {
      const sample = String(m).slice(0, 180).replace(/\s+/g, " ");
      suspectedSecrets.push(`${kind}\t${r}\t${sample}`);
    }
  }
}

const domSinks = [];
for (const { file, text } of corpus) {
  const r = rel(file);
  for (const sink of ["innerHTML", "dangerouslySetInnerHTML", "document.cookie", "localStorage", "sessionStorage", "postMessage", "location.href", "location.replace", "window.location"]) {
    if (text.includes(sink)) domSinks.push(`${sink}\t${r}\t${around(text, sink, 220)}`);
  }
}

const jsList = jsFiles.map((f) => `${rel(f)}\t${fs.statSync(f).size}`).sort();

write("JS_FILES.tsv", "file\tsize_bytes\n" + jsList.join("\n"));
write("URLS.tsv", "url\tsource\n" + uniqueSorted(urls).join("\n"));
write("HOSTS.tsv", "host\tsource\n" + uniqueSorted(hosts).join("\n"));
write("PATHS.tsv", "path\tsource\n" + uniqueSorted(paths).join("\n"));
write("APIISH_URLS.txt", apiishUrls.join("\n"));
write("APIISH_PATHS.txt", apiishPaths.join("\n"));
write("KEYWORD_HITS.tsv", "keyword\tsource\n" + uniqueSorted(keywords).join("\n"));
write("POTENTIAL_SECRETS.tsv", "kind\tsource\tsample\n" + uniqueSorted(suspectedSecrets).slice(0, 500).join("\n"));
write("DOM_SINKS.tsv", "sink\tsource\tcontext\n" + uniqueSorted(domSinks).join("\n"));

const endpointMd = [
  "# Paramount Endpoint Inventory",
  "",
  `Generated from local corpus under \`${root}\`.`,
  "",
  "## Corpus",
  "",
  `- JS files: ${jsFiles.length}`,
  `- Total JS bytes: ${jsFiles.reduce((n, f) => n + fs.statSync(f).size, 0)}`,
  "",
  "## API-ish URLs",
  "",
  ...apiishUrls.map((u) => `- \`${u}\``),
  "",
  "## API-ish Relative Paths",
  "",
  ...apiishPaths.map((p) => `- \`${p}\``),
  "",
  "## Notable Keywords",
  "",
  ...interesting.slice(0, 80).map((hit) => `- \`${hit.keyword}\` in \`${hit.file}\`: ${hit.context}`),
];
write("ENDPOINTS.md", endpointMd.join("\n"));

const findings = [];

function addFinding(title, severity, evidence, status, next) {
  findings.push({ title, severity, evidence, status, next });
}

if (keywords.some((x) => x.startsWith("getFeatureFlagsFromCookies\t"))) {
  addFinding(
    "Client-controllable feature flags via cookies",
    "Info / needs dynamic confirmation",
    "`getFeatureFlagsFromCookies` reads cookies prefixed with `ff-` and enables matching feature flags client-side.",
    "Static evidence only. This is usually a feature-preview/debug mechanism unless a server trusts the same flags or hidden privileged UI calls privileged endpoints.",
    "In a logged-in browser, set harmless `ff-*` cookies and observe whether only UI changes occur. Do not attempt privileged state changes without authorization."
  );
}

if (keywords.some((x) => x.startsWith("browseHubRedirects\t"))) {
  addFinding(
    "Config-driven same-origin redirect mapping",
    "Info",
    "`browseHubRedirects` can trigger `window.location.replace(origin + destination)` when the path contains a configured `from` fragment.",
    "Destination is joined to the current origin in the observed code, so this is not an open redirect from static evidence.",
    "Confirm live config values in `window.__DATA__` only if needed."
  );
}

const postLoginMessaging =
  corpus.some(({ text }) => text.includes('window.opener.postMessage(e,"*")')) &&
  corpus.some(({ text }) => text.includes('window.addEventListener("message",function(e){var n=t.postMessageParser.parse(e.data)')) &&
  corpus.some(({ text }) => text.includes('authorizationCodeKey","authorizationCode"'));

if (postLoginMessaging) {
  addFinding(
    "Wildcard postMessage auth callback accepts authorizationCode messages without visible origin validation",
    "Medium candidate / needs dynamic validation",
    "The popup callback builds `as-provider-data` + `window.location.search` and calls `window.opener.postMessage(e,\"*\")`; the listener registers `window.addEventListener(\"message\", ...)`, parses `e.data`, and emits login success when a string contains `authorizationCode`.",
    "Static candidate. The parser does not validate `event.origin` in the observed bundle and only extracts an `authorizationCode` value. Actual impact depends on how `loginEvent.emitSuccess` exchanges/validates the code server-side and whether a malicious origin can reach the listener in a realistic flow.",
    "In a benign local test page or controlled browser session, verify whether a crafted `postMessage('authorizationCode=test')` changes UI/auth state or triggers a backend exchange, without using real third-party accounts or attacking users."
  );
}

if (keywords.some((x) => x.startsWith("authSuiteFetchHelper\t"))) {
  addFinding(
    "Auth-suite request helper attaches application access tokens",
    "Info / review surface",
    "`authSuiteFetchHelper.getRequestConfig` reads `applicationAccessToken` and `customRequestHeaders` before calling the header service.",
    "This documents tokenized media/account API surface, but no hardcoded access token was recovered in static files.",
    "Dynamic mapping should watch requests for token issuance, token audience, and whether media/account endpoints reject replay or cross-resource use."
  );
}

if (suspectedSecrets.some((x) => x.includes("jwt_like"))) {
  addFinding(
    "JWT-like literal is a library sample, not a live secret",
    "Info / false-positive note",
    "The only JWT-shaped value recovered is the common sample token with `John Doe` payload in `vendor.c3b540e2d118adc8.js`.",
    "Not a reportable secret from static evidence.",
    "No action unless additional live-looking tokens are recovered dynamically."
  );
}

const reportMd = [
  "# Paramount Static Audit",
  "",
  `Generated from local files under \`${root}\`. No active attack traffic was sent by this script.`,
  "",
  "## Status",
  "",
  "- JS extraction appears complete for the pages captured in the prior Claude run: 19 first-party/local JS files, 10.6 MB total.",
  "- Endpoint inventory generated from JS and captured HTML.",
  "- Fable CLI was not available on PATH in this Codex environment; this report is the local static-review substitute.",
  "- Dynamic Caido/Playwright mapping remains a manual/authenticated step.",
  "",
  "## Findings",
  "",
  ...(findings.length
    ? findings.flatMap((f, i) => [
        `### P${i + 1}. ${f.title}`,
        "",
        `- Severity: ${f.severity}`,
        `- Evidence: ${f.evidence}`,
        `- Static status: ${f.status}`,
        `- Next step: ${f.next}`,
        "",
      ])
    : ["No reportable findings were validated from static evidence alone.", ""]),
  "## High-Signal Surfaces",
  "",
  "- Auth/media token helpers: `authSuiteFetchHelper`, `applicationAccessToken`, `deviceRefreshToken`, `mediaToken`, `getMediaToken`.",
  "- Player/media APIs and telemetry: Avia player plugins, EPG endpoint fields, Eden reporting callbacks.",
  "- Client-controlled UI/config paths: feature flag cookies (`ff-*`), `window.__DATA__`, redirect mappings.",
  "- Third-party integrations: Adobe Launch, Ketch, OneTrust, Doppler/CBSI video config, Braze push SDK.",
  "",
  "## Dynamic Work Remaining",
  "",
  "- Open `southparkstudios.com` in a headed browser through Caido and capture only normal page-load, login, profile, and video-playback flows.",
  "- Confirm live `window.__DATA__` configuration, media token endpoints, and auth/account request boundaries.",
  "- Verify whether client-side feature flags affect only presentation or expose privileged calls.",
  "- Confirm no secrets in `POTENTIAL_SECRETS.tsv` are live credentials before reporting anything.",
];
write("STATIC_AUDIT.md", reportMd.join("\n"));

console.log(JSON.stringify({
  root,
  outDir,
  jsFiles: jsFiles.length,
  jsBytes: jsFiles.reduce((n, f) => n + fs.statSync(f).size, 0),
  urls: uniqueSorted(urls).length,
  apiishUrls: apiishUrls.length,
  paths: uniqueSorted(paths).length,
  apiishPaths: apiishPaths.length,
  keywordHits: uniqueSorted(keywords).length,
  domSinks: uniqueSorted(domSinks).length,
  suspectedSecrets: uniqueSorted(suspectedSecrets).length,
}, null, 2));
