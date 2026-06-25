param(
  [string]$Config = "configs/demeter.yml",
  [int]$Jobs = 64,
  [ValidateSet("slow", "normal", "burst")]
  [string]$Mode = "normal",
  [string]$Output = "results/simulation.csv"
)

$ErrorActionPreference = "Stop"
python -m MARL.cli simulate --config $Config --jobs $Jobs --mode $Mode --output $Output
