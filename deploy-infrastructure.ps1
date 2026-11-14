<#
 Deploy script: Face Recognition EC2 instance via Terraform.
 Usage:  .\deploy-infrastructure.ps1

 Flow:   check tools + AWS identity -> terraform init -> validate -> PLAN (saved to a file)
         -> you read the plan and type "yes" -> apply exactly that saved plan -> show outputs.
 Nothing is created until you confirm. There is no -auto-approve.
#>

$ErrorActionPreference = "Stop"

# PowerShell does not stop on a failing native program by itself, so every terraform/aws call goes
# through this helper, which checks $LASTEXITCODE and throws.
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed (exit code $LASTEXITCODE)."
    }
}

Write-Host "`n[1/6] Checking tools and AWS authentication..." -ForegroundColor Cyan

foreach ($tool in @("terraform", "aws")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is not installed or not in your PATH. Please install it first."
    }
}

$identityJson = aws sts get-caller-identity --output json
if ($LASTEXITCODE -ne 0) {
    throw "AWS authentication failed. Configure or verify your AWS CLI credentials (aws sts get-caller-identity)."
}
$callerIdentity = $identityJson | ConvertFrom-Json
Write-Host "  -> Authenticated as: $($callerIdentity.Arn) (Account: $($callerIdentity.Account))" -ForegroundColor Green

$terraformDir = Join-Path -Path $PSScriptRoot -ChildPath "terraform"
Push-Location $terraformDir

try {
    Write-Host "`n[2/6] Initializing Terraform (downloads the AWS provider if needed)..." -ForegroundColor Cyan
    Invoke-Native "terraform init" { terraform init -input=false }

    Write-Host "`n[3/6] Validating Terraform configuration..." -ForegroundColor Cyan
    Invoke-Native "terraform validate" { terraform validate }

    Write-Host "`n[4/6] Planning (nothing is created in this step)..." -ForegroundColor Cyan
    Invoke-Native "terraform plan" { terraform plan -input=false -out=tfplan }

    Write-Host "`n[5/6] Review the plan above." -ForegroundColor Yellow
    $answer = Read-Host "Type 'yes' to create these resources in AWS (anything else cancels)"
    if ($answer -ne "yes") {
        Write-Host "Cancelled. Nothing was created." -ForegroundColor Yellow
        return
    }

    Write-Host "`nApplying the saved plan..." -ForegroundColor Cyan
    Invoke-Native "terraform apply" { terraform apply -input=false tfplan }

    Write-Host "`n[6/6] Reading outputs..." -ForegroundColor Cyan
    $publicIp = terraform output -raw instance_public_ip
    if ($LASTEXITCODE -ne 0) { throw "terraform output instance_public_ip failed (exit code $LASTEXITCODE)." }
    $instanceId = terraform output -raw instance_id
    if ($LASTEXITCODE -ne 0) { throw "terraform output instance_id failed (exit code $LASTEXITCODE)." }

    Write-Host "`n========================================================" -ForegroundColor Green
    Write-Host " EC2 INSTANCE PROVISIONED" -ForegroundColor Green
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host " Instance ID : $instanceId" -ForegroundColor Yellow
    Write-Host " Public IP   : $publicIp" -ForegroundColor Yellow
    Write-Host " SSH Command : ssh -i <your-key-file>.pem ec2-user@$publicIp" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Green
    Write-Host "Next: point DNS at the new IP, update the GitHub secrets EC2_HOST and EC2_KNOWN_HOSTS, then deploy (see CICD-README.md)." -ForegroundColor Cyan
}
finally {
    if (Test-Path "tfplan") { Remove-Item "tfplan" -Force }
    Pop-Location
}
