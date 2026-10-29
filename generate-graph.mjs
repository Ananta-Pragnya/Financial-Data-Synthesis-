import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const [inputArg, outputArg] = process.argv.slice(2);
if (!inputArg) {
  console.error("Usage: node generate-graph.mjs <source.csv> [output-directory]");
  process.exit(2);
}

const inputPath = path.resolve(inputArg);
const outputDir = path.resolve(outputArg || path.join(import.meta.dirname, "import"));
fs.mkdirSync(outputDir, { recursive: true });

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.endsWith("\r") ? field.slice(0, -1) : field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(filename, headers, rows) {
  const content = [headers, ...rows]
    .map((row) => row.map(csvCell).join(","))
    .join("\r\n");
  fs.writeFileSync(path.join(outputDir, filename), `${content}\r\n`, "utf8");
}

function increment(map, key) {
  map.set(key, (map.get(key) || 0) + 1);
}

const raw = fs.readFileSync(inputPath, "utf8").replace(/^\uFEFF/, "");
const matrix = parseCsv(raw);
const headers = matrix.shift();
const expected = [
  "news_date", "ticker_id", "title", "sentiment_label", "impact_score",
  "time_horizon_months", "confidence", "sentiment_reason",
];
if (headers.join("|") !== expected.join("|")) {
  throw new Error(`Unexpected columns: ${headers.join(", ")}`);
}

const records = [];
const companies = new Map();
const sentiments = new Map();
const horizons = new Map();
const months = new Map();
const years = new Map();
const identityOccurrences = new Map();
let missingValues = 0;
let invalidValues = 0;
let minDate = null;
let maxDate = null;
let impactSum = 0;
let confidenceSum = 0;

for (let index = 0; index < matrix.length; index += 1) {
  const values = matrix[index];
  if (values.length === 1 && values[0] === "") continue;
  if (values.length !== headers.length) {
    throw new Error(`Row ${index + 2} has ${values.length} fields; expected ${headers.length}`);
  }
  const row = Object.fromEntries(headers.map((header, column) => [header, values[column].trim()]));
  missingValues += Object.values(row).filter((value) => value === "").length;
  const date = new Date(row.news_date);
  const impact = Number(row.impact_score);
  const confidence = Number(row.confidence);
  const sentiment = row.sentiment_label.toLowerCase();
  if (Number.isNaN(date.valueOf()) || !Number.isFinite(impact) || !Number.isFinite(confidence)) invalidValues += 1;
  if (!new Set(["positive", "negative", "neutral"]).has(sentiment)) invalidValues += 1;
  if (!new Set(["0-1", "1-3", "3-6"]).has(row.time_horizon_months)) invalidValues += 1;
  if (impact < -3 || impact > 3 || confidence < 1 || confidence > 3) invalidValues += 1;

  const iso = date.toISOString();
  const monthId = iso.slice(0, 7);
  const year = Number(iso.slice(0, 4));
  const companyId = `company:${crypto.createHash("sha256").update(row.ticker_id).digest("hex").slice(0, 20)}`;
  const identity = headers.map((header) => row[header]).join("\u0000");
  const occurrence = (identityOccurrences.get(identity) || 0) + 1;
  identityOccurrences.set(identity, occurrence);
  const newsId = `news:${crypto.createHash("sha256").update(`${identity}\u0000${occurrence}`).digest("hex").slice(0, 24)}`;

  companies.set(companyId, row.ticker_id);
  increment(sentiments, sentiment);
  increment(horizons, row.time_horizon_months);
  increment(months, monthId);
  increment(years, String(year));
  minDate = minDate == null || iso < minDate ? iso : minDate;
  maxDate = maxDate == null || iso > maxDate ? iso : maxDate;
  impactSum += impact;
  confidenceSum += confidence;
  records.push({ ...row, newsId, companyId, sentiment, iso, monthId, year, impact, confidence, sourceRow: index + 2 });
}

const companyCounts = new Map();
for (const record of records) increment(companyCounts, record.ticker_id);

writeCsv("dataset.csv", ["id:ID(Dataset-ID)", "name", "description", "license", "row_count:int", "start_date:datetime", "end_date:datetime"], [[
  "dataset:dax40_news_sentiment_2017_2026",
  "DAX40 Financial News Sentiment",
  "LLM-labeled financial news headlines for DAX40 constituents",
  "Non-commercial research only",
  records.length,
  minDate,
  maxDate,
]]);
writeCsv("companies.csv", ["id:ID(Company-ID)", "name"], [...companies].sort((a, b) => a[1].localeCompare(b[1])).map(([id, name]) => [id, name]));
writeCsv("sentiments.csv", ["id:ID(Sentiment-ID)", "name"], ["negative", "neutral", "positive"].map((name) => [`sentiment:${name}`, name]));
writeCsv("horizons.csv", ["id:ID(Horizon-ID)", "name", "min_months:int", "max_months:int"], [["horizon:0-1", "0-1", 0, 1], ["horizon:1-3", "1-3", 1, 3], ["horizon:3-6", "3-6", 3, 6]]);
writeCsv("months.csv", ["id:ID(Month-ID)", "label", "year:int", "month:int"], [...months.keys()].sort().map((id) => [`month:${id}`, id, Number(id.slice(0, 4)), Number(id.slice(5, 7))]));
writeCsv("news.csv", ["id:ID(News-ID)", "news_date:datetime", "title", "impact_score:double", "confidence:int", "sentiment_reason", "source_row:int"], records.map((r) => [r.newsId, r.iso, r.title, r.impact, r.confidence, r.sentiment_reason, r.sourceRow]));
writeCsv("news_about_company.csv", [":START_ID(News-ID)", ":END_ID(Company-ID)", ":TYPE"], records.map((r) => [r.newsId, r.companyId, "ABOUT"]));
writeCsv("news_has_sentiment.csv", [":START_ID(News-ID)", ":END_ID(Sentiment-ID)", ":TYPE"], records.map((r) => [r.newsId, `sentiment:${r.sentiment}`, "HAS_SENTIMENT"]));
writeCsv("news_has_horizon.csv", [":START_ID(News-ID)", ":END_ID(Horizon-ID)", ":TYPE"], records.map((r) => [r.newsId, `horizon:${r.time_horizon_months}`, "HAS_HORIZON"]));
writeCsv("news_published_in.csv", [":START_ID(News-ID)", ":END_ID(Month-ID)", ":TYPE"], records.map((r) => [r.newsId, `month:${r.monthId}`, "PUBLISHED_IN"]));
writeCsv("dataset_contains_news.csv", [":START_ID(Dataset-ID)", ":END_ID(News-ID)", ":TYPE"], records.map((r) => ["dataset:dax40_news_sentiment_2017_2026", r.newsId, "CONTAINS"]));

const sortCounts = (map) => Object.fromEntries([...map].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])));
const stats = {
  source: inputPath,
  generated_at: new Date().toISOString(),
  rows: records.length,
  companies: companies.size,
  date_range: { start: minDate, end: maxDate },
  nodes: 1 + companies.size + 3 + 3 + months.size + records.length,
  relationships: records.length * 5,
  missing_values: missingValues,
  invalid_values: invalidValues,
  exact_duplicate_rows: [...identityOccurrences.values()].reduce((sum, count) => sum + Math.max(0, count - 1), 0),
  average_impact_score: Number((impactSum / records.length).toFixed(4)),
  average_confidence: Number((confidenceSum / records.length).toFixed(4)),
  sentiment_counts: sortCounts(sentiments),
  horizon_counts: sortCounts(horizons),
  year_counts: Object.fromEntries([...years].sort((a, b) => a[0].localeCompare(b[0]))),
  company_counts: sortCounts(companyCounts),
};
fs.writeFileSync(path.join(path.dirname(outputDir), "analysis.json"), `${JSON.stringify(stats, null, 2)}\n`, "utf8");
console.log(JSON.stringify(stats, null, 2));
