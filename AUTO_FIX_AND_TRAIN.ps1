# AUTO_FIX_AND_TRAIN.ps1
# سكريبت إصلاح تلقائي كامل

param(
    [string]$ProjectPath = "E:\OCR_system-Atlas"
)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Auto Fix and Train - VideoX Action Recognition" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# التأكد من المسار
if (-not (Test-Path $ProjectPath)) {
    Write-Host "❌ Project path not found: $ProjectPath" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectPath

Write-Host "[1/4] Backing up original files..." -ForegroundColor Yellow

# نسخ احتياطي
$backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

Copy-Item "src\model_architecture.py" "$backupDir\model_architecture.py.bak" -ErrorAction SilentlyContinue
Copy-Item "src\trainer_module.py" "$backupDir\trainer_module.py.bak" -ErrorAction SilentlyContinue

Write-Host "  ✅ Backup created in $backupDir" -ForegroundColor Green

Write-Host ""
Write-Host "[2/4] Fixing model_architecture.py..." -ForegroundColor Yellow

# قراءة الملف
$modelFile = "src\model_architecture.py"
$content = Get-Content $modelFile -Raw

# البحث والاستبدال - إزالة Sigmoid من boundary_detector
$pattern = '(self\.boundary_detector\s*=\s*nn\.Sequential\s*\(\s*nn\.Linear[^)]+\),\s*nn\.GELU\(\),\s*nn\.Dropout[^)]+\),\s*nn\.Linear[^)]+\)),\s*nn\.Sigmoid\(\)'
$replacement = '$1'

if ($content -match 'nn\.Sigmoid\(\)\s*#.*boundary') {
    $content = $content -replace ',\s*nn\.Sigmoid\(\)\s*#.*boundary.*\n', ''
    Write-Host "  ✅ Removed Sigmoid from boundary_detector" -ForegroundColor Green
}
elseif ($content -match 'nn\.Sigmoid\(\)\s*\)\s*#.*Boundary') {
    $content = $content -replace ',\s*nn\.Sigmoid\(\)\s*\n\s*\)\s*#.*Boundary', '`n        ) # Boundary'
    Write-Host "  ✅ Removed Sigmoid from boundary_detector" -ForegroundColor Green
}
else {
    # بحث بسيط
    $lines = $content -split "`n"
    $newLines = @()
    $inBoundaryDetector = $false
    
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        
        if ($line -match 'self\.boundary_detector') {
            $inBoundaryDetector = $true
        }
        
        if ($inBoundaryDetector -and $line -match 'nn\.Sigmoid\(\)') {
            Write-Host "  ⚠️  Found Sigmoid at line $($i+1), skipping..." -ForegroundColor Yellow
            # تخطي هذا السطر
            continue
        }
        
        $newLines += $line
        
        if ($inBoundaryDetector -and $line -match '^\s*\)') {
            $inBoundaryDetector = $false
        }
    }
    
    $content = $newLines -join "`n"
    Write-Host "  ✅ Processed model_architecture.py" -ForegroundColor Green
}

# حفظ الملف
$content | Set-Content $modelFile -NoNewline

Write-Host ""
Write-Host "[3/4] Checking trainer_module.py..." -ForegroundColor Yellow

$trainerFile = "src\trainer_module.py"
$trainerContent = Get-Content $trainerFile -Raw

# التحقق من الإصلاحات
$needsFix = $false

if ($trainerContent -match 'from torch\.cuda\.amp import') {
    Write-Host "  ⚠️  Old import found: torch.cuda.amp" -ForegroundColor Yellow
    $needsFix = $true
}

if ($trainerContent -match 'nn\.BCELoss\(\)') {
    Write-Host "  ⚠️  Found BCELoss (should be BCEWithLogitsLoss)" -ForegroundColor Yellow
    $needsFix = $true
}

if ($needsFix) {
    Write-Host "  ℹ️  trainer_module.py needs manual fix" -ForegroundColor Cyan
    Write-Host "     Please copy trainer_module_fixed.py to src\trainer_module.py" -ForegroundColor Cyan
}
else {
    Write-Host "  ✅ trainer_module.py looks good" -ForegroundColor Green
}

Write-Host ""
Write-Host "[4/4] Ready to train!" -ForegroundColor Yellow
Write-Host ""

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Verify the fixes:" -ForegroundColor White
Write-Host "   - src\model_architecture.py (Sigmoid removed)" -ForegroundColor Gray
Write-Host "   - src\trainer_module.py (BCEWithLogitsLoss)" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Train the model:" -ForegroundColor White
Write-Host "   python main.py --mode train --epochs 20" -ForegroundColor Green
Write-Host ""
Write-Host "3. Start the server:" -ForegroundColor White
Write-Host "   python app.py" -ForegroundColor Green
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Press any key to start training now..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# تفعيل البيئة والتدريب
& "venv\Scripts\Activate.ps1"
python main.py --mode train --epochs 20
