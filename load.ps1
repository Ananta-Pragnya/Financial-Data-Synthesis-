$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $project
try {
    docker compose up -d --build
    Write-Host 'Waiting for Neo4j health check...'
    $healthy = $false
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        $status = docker inspect --format '{{.State.Health.Status}}' dax40-knowledge-graph-neo4j 2>$null
        if ($status -eq 'healthy') { $healthy = $true; break }
        Start-Sleep -Seconds 5
    }
    if (-not $healthy) { throw 'Neo4j did not become healthy within 150 seconds.' }
    docker compose exec -T knowledge-graph-db cypher-shell -u neo4j -p 12345678 -f /cypher/load.cypher
    docker compose exec -T knowledge-graph-db cypher-shell -u neo4j -p 12345678 -f /cypher/verify.cypher
    Write-Host 'Neo4j Browser: http://localhost:7474 (neo4j / 12345678)'
} finally {
    Pop-Location
}

