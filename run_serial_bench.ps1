# run_serial_bench.ps1
# Usage: run this in the folder with DBSCAN_serial.exe and the *_data.csv files.
# It creates serial_raw.csv (per-iteration) and serial_summary.csv (aggregates).
# NOTE: Measures total runtime of each DBSCAN_serial.exe invocation (includes I/O).
# To measure compute-only you must add a compute-only/bench mode to the binary and
# call that mode instead.

# --- Config
$datasets = @(20000,40000,80000,120000)
$eps = 0.03
$minSamples = 10
$iterations = 10
$serialExe = ".\DBSCAN_serial.exe"
$outRaw = "serial_raw.csv"
$outSummary = "serial_summary.csv"
$threads = 1  # serial runs -> threads=1

# --- Prepare output
"n_points,threads,iteration,elapsed_ms,cores1" | Out-File -Encoding utf8 $outRaw

foreach ($n in $datasets) {
    $input = "$($n)_data.csv"
    if (-not (Test-Path $input)) {
        Write-Warning "Skipping $input (file not found)"
        continue
    }

    Write-Host ""
    Write-Host "=== Dataset $($n): warm-up run ==="
    # Warm-up run (ignore timing), suppress output
    & $serialExe $input $eps $minSamples > $null 2> $null

    for ($it = 1; $it -le $iterations; $it++) {
    Write-Host "Dataset $($n) - iteration $($it) ..."

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        # Run the serial binary; suppress its stdout/stderr so log is clean.
        & $serialExe $input $eps $minSamples > $null 2> $null
        $sw.Stop()
        $ms = $sw.Elapsed.TotalMilliseconds

        # Parse output results file to count cores labeled '1'
        $resultsFile = "$($n)_results.csv"
        $cores1 = 0
        if (Test-Path $resultsFile) {
            # Read lines, skip empty lines, parse third column (type)
            Get-Content $resultsFile | ForEach-Object {
                $line = $_.Trim()
                if ($line -eq "") { return }
                # split on comma, but be tolerant of spaces
                $parts = $line -split ','
                if ($parts.Length -ge 3) {
                    $typeStr = $parts[2].Trim()
                    # Some files might have non-numeric trailing; try parse as int
                    $typeInt = 0
                    if ([int]::TryParse($typeStr, [ref]$typeInt)) {
                        if ($typeInt -eq 1) { $cores1++ }
                    }
                }
            }
        } else {
            Write-Warning "Results file $resultsFile not found after run; cores1 set to 0"
        }

        # Append to raw CSV
        "$n,$threads,$it,$([math]::Round($ms,6)),$cores1" | Out-File -Append -Encoding utf8 $outRaw
        Write-Host ("  -> ms={0:N3}, cores1={1}" -f $ms, $cores1)
    }
}

# --- Aggregate summary: compute mean and stddev per n_points
# Read raw CSV and group by n_points
$rows = Import-Csv $outRaw
$groups = $rows | Group-Object -Property n_points

"n_points,threads,mean_ms,stddev_ms,mean_cores1" | Out-File -Encoding utf8 $outSummary

foreach ($g in $groups) {
    $n = [int]$g.Name
    $vals = $g.Group | ForEach-Object { [double]$_.elapsed_ms }
    $count = $vals.Count
    if ($count -eq 0) { continue }
    $mean = ($vals | Measure-Object -Average).Average
    # population stddev (divide by N). If you prefer sample stddev use (N-1)
    $sumSquares = ($vals | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Sum).Sum
    $stddev = [math]::Sqrt($sumSquares / [math]::Max(1,$count))
    $meanCores = ($g.Group | ForEach-Object { [int]$_.cores1 } | Measure-Object -Average).Average

    "$n,1,$([math]::Round($mean,6)),$([math]::Round($stddev,6)),$([math]::Round($meanCores,3))" | Out-File -Append -Encoding utf8 $outSummary
    Write-Host ("Summary N={0}: mean_ms={1:N3}, stddev_ms={2:N3}, mean_cores1={3:N0}" -f $n, $mean, $stddev, $meanCores)
}

Write-Host "`nDone. Raw: $outRaw  Summary: $outSummary"