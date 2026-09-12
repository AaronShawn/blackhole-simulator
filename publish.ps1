<#
  publish.ps1 - create the GitHub repository and push this project.

  Usage (PowerShell):

      # token from the environment (recommended)
      $env:GITHUB_TOKEN = "ghp_xxx"       # classic PAT with 'repo' scope,
      .\publish.ps1                       # or fine-grained with Contents+Metadata
                                          # + Administration: Read and write

      # or let the script ask (input is hidden)
      .\publish.ps1 -RepoName blackhole-simulator -Visibility public

  What it does:
    1. authenticates against api.github.com
    2. fills the __OWNER__ / __REPO__ placeholders in README/CONTRIBUTING/CITATION
    3. sets the git identity from your GitHub account (noreply e-mail)
    4. creates the repository (skips if it already exists)
    5. configures the proxy (when the system proxy is on) and pushes 'main'

  Nothing is uploaded until step 5 - the script prints the repository URL at the end.
#>
[CmdletBinding()]
param(
    [string]$RepoName = "blackhole-simulator",
    [string]$Description = "Real-time Schwarzschild black hole ray tracer - exact null geodesics, relativistic accretion disk, OrbitControls, portable Windows build (Three.js + WebGL2 + GLSL).",
    [ValidateSet("public", "private")][string]$Visibility = "public",
    [string]$Owner = "",
    [string]$Token = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-SystemProxy {
    try {
        $p = Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction Stop
        if ($p.ProxyEnable -eq 1 -and $p.ProxyServer) { return "http://$($p.ProxyServer)" }
    } catch { }
    return $null
}

function Invoke-GitHub {
    param(
        [string]$Method = "GET",
        [string]$Uri,
        [object]$Body = $null,
        [hashtable]$Headers
    )
    $splat = @{ Method = $Method; Uri = $Uri; Headers = $Headers; ContentType = "application/json" }
    if ($Body) { $splat.Body = ($Body | ConvertTo-Json -Depth 6) }
    if ($script:ProxyUri) { $splat.Proxy = $script:ProxyUri }
    return Invoke-RestMethod @splat
}

# ---------------------------------------------------------------- credentials
$script:ProxyUri = Get-SystemProxy
if (-not $Token) { $Token = $env:GITHUB_TOKEN }
if (-not $Token) { $Token = $env:GH_TOKEN }
if (-not $Token) {
    Write-Host "Paste a GitHub Personal Access Token (input hidden)." -ForegroundColor Yellow
    Write-Host "  classic token needs scope: repo (and workflow if you edit CI)"
    Write-Host "  fine-grained token needs: Contents RW, Administration RW, Metadata R"
    $secure = Read-Host -AsSecureString "Token"
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $Token = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}
if (-not $Token) { throw "No token supplied." }

$headers = @{
    Authorization          = "Bearer $Token"
    Accept                 = "application/vnd.github+json"
    "User-Agent"           = "blackhole-simulator-publish"
    "X-GitHub-Api-Version" = "2022-11-28"
}

Write-Host "[1/5] authenticating ..."
try {
    $me = Invoke-GitHub -Uri "https://api.github.com/user" -Headers $headers
} catch {
    if ($script:ProxyUri) { throw "Authentication failed even through proxy $script:ProxyUri : $_" }
    $script:ProxyUri = $null
    throw "Authentication failed: $_"
}
$login = $me.login
$owner = if ($Owner) { $Owner } else { $login }
Write-Host ("      logged in as {0} ({1}); target: {2}/{3} [{4}]" -f $login, $me.name, $owner, $RepoName, $Visibility)

# ------------------------------------------------------------------ placeholders
Write-Host "[2/5] filling __OWNER__ / __REPO__ placeholders ..."
$targets = @("README.md", "README.en.md", "CONTRIBUTING.md", "CITATION.cff")
foreach ($f in $targets) {
    if (-not (Test-Path $f)) { continue }
    $text = Get-Content -LiteralPath $f -Raw -Encoding UTF8
    $new = $text.Replace("__OWNER__", $owner).Replace("__REPO__", $RepoName)
    if ($new -ne $text) {
        [IO.File]::WriteAllText((Resolve-Path $f), $new, (New-Object Text.UTF8Encoding $false))
        Write-Host "      updated $f"
    }
}

# ------------------------------------------------------------------ git commit
Write-Host "[3/5] committing ..."
if (-not (Test-Path ".git")) { git init -b main | Out-Null }
git config user.name  $login | Out-Null
git config user.email "$login@users.noreply.github.com" | Out-Null
git add -A
$pending = git status --porcelain
if ($pending) {
    if (git rev-parse --verify HEAD 2>$null) {
        git commit --amend --no-edit --reset-author | Out-Null
    } else {
        git commit -m "Schwarzschild black hole simulator v1.0.0" | Out-Null
    }
    Write-Host "      committed"
} else {
    Write-Host "      nothing to commit"
}

if ($DryRun) {
    Write-Host "[dry-run] skipping repository creation and push."
    return
}

# ------------------------------------------------------------------ create repo
Write-Host "[4/5] ensuring repository $owner/$RepoName ..."
$exists = $false
try {
    Invoke-GitHub -Uri "https://api.github.com/repos/$owner/$RepoName" -Headers $headers | Out-Null
    $exists = $true
    Write-Host "      repository already exists"
} catch { }

if (-not $exists) {
    $payload = @{
        name        = $RepoName
        description = $Description
        private     = ($Visibility -eq "private")
        has_issues  = $true
        has_wiki    = $false
        auto_init   = $false
    }
    $uri = if ($owner -eq $login) { "https://api.github.com/user/repos" } else { "https://api.github.com/orgs/$owner/repos" }
    $repo = Invoke-GitHub -Method POST -Uri $uri -Body $payload -Headers $headers
    Write-Host "      created: $($repo.html_url)"
}

# ------------------------------------------------------------------------ push
Write-Host "[5/5] pushing ..."
$remoteUrl = "https://github.com/$owner/$RepoName.git"
$existing = git remote 2>$null
if ($existing -contains "origin") { git remote set-url origin $remoteUrl } else { git remote add origin $remoteUrl }

if ($script:ProxyUri) {
    git config --local http.proxy  $script:ProxyUri | Out-Null
    git config --local https.proxy $script:ProxyUri | Out-Null
    Write-Host "      using proxy $script:ProxyUri for git"
}

# keep the token out of .git/config: temporary credential store, removed afterwards
$credFile = Join-Path ([IO.Path]::GetTempPath()) ("bh-" + [guid]::NewGuid().ToString("N") + ".cred")
$escaped = $Token -replace "@", "%40"
[IO.File]::WriteAllText($credFile, "https://x-access-token:$escaped@github.com`n", (New-Object Text.ASCIIEncoding))
try {
    git -c "credential.helper=store --file=$credFile" push -u origin main
    if ($LASTEXITCODE -ne 0) { throw "git push failed (exit $LASTEXITCODE)" }
} finally {
    if (Test-Path $credFile) { Remove-Item -LiteralPath $credFile -Force }
}

Write-Host ""
Write-Host "Done -> $remoteUrl" -ForegroundColor Green
Write-Host "Next ideas:"
Write-Host "  * tag a release:  git tag v1.0.0; git push origin v1.0.0   (CI builds the portable zip)"
Write-Host "  * add topics:     black-hole, general-relativity, webgl, threejs, glsl, raytracing"
Write-Host "  * pin the repo and add the screenshots from docs/ to the description"
