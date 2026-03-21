$ErrorActionPreference = "Stop"

$nbPath = "notebooks\01_Exploratory_Data_Analysis.ipynb"
if (-not (Test-Path $nbPath)) {
    Write-Error "Notebook not found at $nbPath"
}

$nb = Get-Content $nbPath -Raw | ConvertFrom-Json

$header_md = @{
    cell_type = "markdown"
    metadata = @{}
    source = @(
        "# 🥕 Feed-to-Farm: Exploratory Data Analysis (EDA)`n",
        "`n",
        "Welcome to the Exploratory Data Analysis (EDA) and Feature Engineering notebook for the Feed-to-Farm fresh produce demand challenge.`n",
        "`n",
        "### ⚠️ Execution Warning`n",
        "**To avoid high CPU/RAM usage on your local machine, it is highly recommended to run this notebook in a cloud environment:**`n",
        "- [Open in Google Colab](https://colab.research.google.com/)`n",
        "- [Open in Kaggle Kernels](https://www.kaggle.com/)`n",
        "`n",
        "The cell below will automatically download the dataset directly from this project's GitHub repository so it works seamlessly in the cloud without manual uploads."
    )
}

$cloud_setup_code = @{
    cell_type = "code"
    execution_count = $null
    metadata = @{}
    outputs = @()
    source = @(
        "# Run this cell if executing on Google Colab or Kaggle to fetch the data directly!`n",
        "import os`n",
        "import urllib.request`n",
        "`n",
        "base_url = 'https://raw.githubusercontent.com/clementina-tom/Feed-to-farm-competition/main/'`n",
        "files = ['Train.csv', 'Test.csv', 'customer_data.csv', 'sku_data.csv']`n",
        "`n",
        "if not os.path.exists('Train.csv'):`n",
        "    print('Downloading data files from GitHub...')`n",
        "    for file in files:`n",
        "        print(f'Fetching {file}...')`n",
        "        try:`n",
        "            urllib.request.urlretrieve(base_url + file, file)`n",
        "        except Exception as e:`n",
        "            print(f'Error fetching {file}: {e}')`n",
        "    print('Data downloaded successfully!')`n",
        "else:`n",
        "    print('Data files already exist locally.')`n"
    )
}

$stop_md = @{
    cell_type = "markdown"
    metadata = @{}
    source = @(
        "---`n",
        "## 🚨 STOP: Heavy Machine Learning Execution Below 🚨`n",
        "The following cells execute the **Hybrid Grandmaster Ensemble (LGBM + CatBoost)** across 5 random seeds.`n",
        "This process takes significant compute time and memory (16GB+ RAM recommended).`n",
        "**Do not run these cells locally unless you have sufficient hardware or are in a cloud environment.**"
    )
}

$new_cells = [System.Collections.ArrayList]::new()
[void]$new_cells.Add($header_md)
[void]$new_cells.Add($cloud_setup_code)

$insertedStop = $false

foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq 'markdown' -and $cell.source -match 'Memory-Efficient ML Pipeline') {
        # Skip the original title cell
        continue
    }

    $sourceText = $cell.source -join ""
    
    if (-not $insertedStop -and $cell.cell_type -eq 'code' -and ($sourceText -match 'lgb\.train' -or $sourceText -match 'CatBoostRegressor')) {
        [void]$new_cells.Add($stop_md)
        $insertedStop = $true
    }
    [void]$new_cells.Add($cell)
}

if (-not $insertedStop) {
    [void]$new_cells.Add($stop_md)
}

$nb.cells = $new_cells

$nb | ConvertTo-Json -Depth 100 > $nbPath
Write-Host "Notebook formatting successful."
