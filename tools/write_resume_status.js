const fs = require("fs");
const path = require("path");

const projectRoot = process.argv[2];
if (!projectRoot) {
  console.error("usage: node tools/write_resume_status.js <claude-project-root>");
  process.exit(2);
}

const status = `# Resume Status

Updated by Codex after picking up Claude session d94789f7-368a-47f9-9f6a-69d49d90a021.

## Completed

- Censys JS extraction:
  - accounts.censys.io: 66 JS files
  - platform.censys.io: 312 JS files
- Censys endpoint/docs/diff notes:
  - censys/notes/ENDPOINTS.md
  - censys/notes/DOCS_VS_JS.md
  - censys/notes/DYNAMIC_MAP.md
- Censys deep static audit summary recovered from Claude temp task output:
  - censys/notes/STATIC_AUDIT_SUMMARY.md
- Paramount/South Park JS extraction from captured pages:
  - paramount/js-recon: 19 JS files, 10.6 MB
- Paramount endpoint/static artifacts:
  - paramount/notes/ENDPOINTS.md
  - paramount/notes/STATIC_AUDIT.md
  - paramount/notes/APIISH_URLS.txt
  - paramount/notes/APIISH_PATHS.txt
  - paramount/notes/HOSTS.tsv
  - paramount/notes/PATHS.tsv
  - paramount/notes/URLS.tsv
  - paramount/notes/DOM_SINKS.tsv
  - paramount/notes/KEYWORD_HITS.tsv
  - paramount/notes/POTENTIAL_SECRETS.tsv

## Highest-Signal Static Candidate

- Paramount auth-flow postMessage candidate:
  - callback sends authorization data to window.opener with target "*"
  - listener parses message data for authorizationCode and emits login success
  - no event.origin check was visible in the static bundle
  - dynamic validation is required before reporting

## Still Requires Dynamic/Auth Work

- Censys dynamic understanding through Caido with user-authenticated Chrome.
- Paramount headed browser/Caido pass for normal login/profile/video playback flows.
- Confirm whether Paramount postMessage candidate changes auth state or triggers backend exchange.
- Confirm whether client-side feature-flag cookies affect only presentation.
- Confirm live token/key values and restrictions from runtime window.__DATA__/window.ENV before reporting.

## Tool Notes

- Fable CLI was not available on PATH in the Codex environment; Paramount STATIC_AUDIT.md is the local static-review substitute.
- No active attack traffic was sent by the Codex static scripts.
`;

const outFile = path.join(projectRoot, "RESUME_STATUS.md");
fs.writeFileSync(outFile, status);
console.log(JSON.stringify({ outFile }, null, 2));
