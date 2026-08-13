const fs = require("fs");
const path = require("path");

const input = process.argv[2];
const outFile = process.argv[3];

if (!input || !outFile) {
  console.error("usage: node tools/censys_task_summary.js <task-output-json> <out-md>");
  process.exit(2);
}

const data = JSON.parse(fs.readFileSync(input, "utf8"));
const confirmed = data.result?.confirmed || [];

function sev(f) {
  return f.verdict?.adjusted_severity || f.severity || "unknown";
}

function countBy(items, fn) {
  const out = {};
  for (const item of items) {
    const k = fn(item);
    out[k] = (out[k] || 0) + 1;
  }
  return out;
}

function tableRows(obj) {
  return Object.keys(obj)
    .sort()
    .map((k) => `| ${k} | ${obj[k]} |`)
    .join("\n");
}

const lines = [
  "# Censys Deep Static Audit Summary",
  "",
  `Source task output: \`${input}\``,
  "",
  "## Run Summary",
  "",
  `- Summary: ${data.summary}`,
  `- Agent count: ${data.agentCount}`,
  `- Raw candidates: ${data.result?.rawCount}`,
  `- Deduped candidates: ${data.result?.dedupedCount}`,
  `- Confirmed after verification: ${data.result?.confirmedCount}`,
  "",
  "## Severity Distribution",
  "",
  "| Severity | Count |",
  "| --- | ---: |",
  tableRows(countBy(confirmed, sev)),
  "",
  "## Category Distribution",
  "",
  "| Category | Count |",
  "| --- | ---: |",
  tableRows(countBy(confirmed, (f) => f.category || "unknown")),
  "",
  "## Confirmed Candidate Index",
  "",
  ...confirmed.map((f, i) => {
    const verdictReason = f.verdict?.verdict_reason
      ? `\n  - Verdict note: ${String(f.verdict.verdict_reason).slice(0, 500)}${String(f.verdict.verdict_reason).length > 500 ? "..." : ""}`
      : "";
    const needsDynamic = f.verdict?.needs_dynamic_confirm ?? f.needs_dynamic_confirm;
    return [
      `### ${i + 1}. ${f.title}`,
      "",
      `- App: ${f.app || "n/a"}`,
      `- Category: ${f.category || "n/a"}`,
      `- Severity: ${sev(f)}`,
      `- Source: \`${f.file || "n/a"}\``,
      `- Endpoint/surface: ${f.endpoint || "n/a"}`,
      `- Needs dynamic confirmation: ${needsDynamic === undefined ? "unknown" : needsDynamic}`,
      verdictReason,
      "",
    ].join("\n");
  }),
  "## Remaining Dynamic Work",
  "",
  "- Re-check every medium/high candidate through normal authenticated browsing only.",
  "- Preserve Caido evidence for server responses before promoting static candidates to reports.",
  "- Treat client-only checks as unconfirmed until the backend authorization decision is observed.",
];

fs.mkdirSync(path.dirname(outFile), { recursive: true });
fs.writeFileSync(outFile, lines.join("\n"));
console.log(JSON.stringify({ outFile, confirmed: confirmed.length }, null, 2));
