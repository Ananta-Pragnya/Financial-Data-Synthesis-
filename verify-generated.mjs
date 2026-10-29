import fs from "node:fs";
import path from "node:path";

const importDir = path.resolve(process.argv[2] || path.join(import.meta.dirname, "import"));

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n") { row.push(field.endsWith("\r") ? field.slice(0, -1) : field); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

const read = (name) => parseCsv(fs.readFileSync(path.join(importDir, name), "utf8"));
const nodeSpecs = [
  ["dataset.csv", "Dataset-ID"], ["companies.csv", "Company-ID"], ["sentiments.csv", "Sentiment-ID"],
  ["horizons.csv", "Horizon-ID"], ["months.csv", "Month-ID"], ["news.csv", "News-ID"],
];
const ids = new Map();
const counts = {};
for (const [file, group] of nodeSpecs) {
  const rows = read(file);
  const data = rows.slice(1).filter((row) => row.length > 1 || row[0]);
  const set = new Set(data.map((row) => row[0]));
  if (set.size !== data.length) throw new Error(`${file}: duplicate node IDs`);
  ids.set(group, set);
  counts[file] = data.length;
}

const relSpecs = [
  ["news_about_company.csv", "News-ID", "Company-ID"],
  ["news_has_sentiment.csv", "News-ID", "Sentiment-ID"],
  ["news_has_horizon.csv", "News-ID", "Horizon-ID"],
  ["news_published_in.csv", "News-ID", "Month-ID"],
  ["dataset_contains_news.csv", "Dataset-ID", "News-ID"],
];
for (const [file, startGroup, endGroup] of relSpecs) {
  const rows = read(file).slice(1).filter((row) => row.length > 1 || row[0]);
  for (let i = 0; i < rows.length; i += 1) {
    if (!ids.get(startGroup).has(rows[i][0])) throw new Error(`${file}:${i + 2}: missing start node ${rows[i][0]}`);
    if (!ids.get(endGroup).has(rows[i][1])) throw new Error(`${file}:${i + 2}: missing end node ${rows[i][1]}`);
  }
  counts[file] = rows.length;
}

const news = counts["news.csv"];
for (const [file] of relSpecs) {
  if (counts[file] !== news) throw new Error(`${file}: expected ${news} relationships, found ${counts[file]}`);
}
console.log(JSON.stringify({ status: "ok", ...counts }, null, 2));

