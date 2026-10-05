$ErrorActionPreference = 'Stop'
$part4WorkbookPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../experiments.xlsx'))
$part4Excel = $null
$part4Workbook = $null
try {
    $part4Excel = New-Object -ComObject Excel.Application
    $part4Excel.Visible = $false
    $part4Excel.DisplayAlerts = $false
    $part4Workbook = $part4Excel.Workbooks.Open($part4WorkbookPath)
    $part4Workbook.Worksheets.Item('Summary').Range('H11').Value2 = 'Baseline lr trials and separate baseline replay; only replay has eval.'
    $part4Excel.Calculation = -4105
    $part4Excel.CalculateFullRebuild()
    $part4Workbook.CheckCompatibility = $false
    $part4Workbook.Save()
    Write-Output "Excel recalculated and saved: $part4WorkbookPath"
}
finally {
    if ($null -ne $part4Workbook) {
        $part4Workbook.Close($false)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($part4Workbook)
    }
    if ($null -ne $part4Excel) {
        $part4Excel.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($part4Excel)
    }
}
