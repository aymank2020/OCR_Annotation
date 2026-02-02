# Quick Setup Script - Copy All Files
# Run this after downloading all files

param(
    [string]$SourceDir = ".",
    [string]$TargetDir = "E:\OCR_system-Atlas"
)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Quick File Setup - VideoX Action Recognition" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Files to copy
$files = @{
    # Root files
    "app.py" = ""
    "main.py" = ""
    "test_system.py" = ""
    "requirements.txt" = ""
    
    # Config
    "config.yaml" = "config\"
    
    # Src files
    "model_architecture.py" = "src\"
    "dataset_loader.py" = "src\"
    "trainer_module.py" = "src\"
    "inference_module.py" = "src\"
    "text_processor.py" = "src\"
}

$copied = 0
$failed = 0

foreach ($file in $files.Keys) {
    $sourcePath = Join-Path $SourceDir $file
    $targetPath = Join-Path $TargetDir $files[$file] $file
    
    Write-Host "Copying: $file" -NoNewline
    
    if (Test-Path $sourcePath) {
        try {
            # Create directory if needed
            $targetFolder = Split-Path $targetPath -Parent
            if (-not (Test-Path $targetFolder)) {
                New-Item -ItemType Directory -Path $targetFolder -Force | Out-Null
            }
            
            # Copy file
            Copy-Item -Path $sourcePath -Destination $targetPath -Force
            Write-Host " ✅" -ForegroundColor Green
            $copied++
        }
        catch {
            Write-Host " ❌ Error: $_" -ForegroundColor Red
            $failed++
        }
    }
    else {
        Write-Host " ⚠️  Not found in source" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  Copied: $copied files" -ForegroundColor Green
Write-Host "  Failed: $failed files" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

if ($copied -gt 0) {
    Write-Host "✅ Files copied successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Edit config\config.yaml - add annotations for your videos" -ForegroundColor White
    Write-Host "  2. Make sure videos are in data\videos\" -ForegroundColor White
    Write-Host "  3. Run: python test_system.py" -ForegroundColor White
    Write-Host "  4. Run: python main.py --mode train --epochs 10" -ForegroundColor White
    Write-Host "  5. Run: python app.py" -ForegroundColor White
    Write-Host ""
}
else {
    Write-Host "⚠️  No files were copied. Check source directory." -ForegroundColor Yellow
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
