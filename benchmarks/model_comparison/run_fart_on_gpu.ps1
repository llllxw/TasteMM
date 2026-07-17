param(
    [string]$Python = "python",
    [string]$Manifest = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$Manifest = if ($Manifest) { $Manifest } else { Join-Path $PSScriptRoot "manifests/six_class_split_manifest.csv" }
$Output = if ($Output) { $Output } else { Join-Path $PSScriptRoot "outputs/fart_augmented" }
$env:USE_TF = "0"
& $Python (Join-Path $PSScriptRoot "run_fart_sixclass.py") `
    --manifest $Manifest `
    --output $Output `
    --variant augmented `
    --epochs 2 `
    --augment-factor 10 `
    --inference-augmentations 10 `
    --batch-size 16 `
    --folds 0 1 2 3 4 `
    --resume
