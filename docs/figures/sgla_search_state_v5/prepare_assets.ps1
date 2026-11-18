param([string]$AssetDirectory=(Join-Path $PSScriptRoot 'assets'))
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$sourcePath=Join-Path $AssetDirectory 'visdrone_street_source.jpg'
$source=[Drawing.Image]::FromFile($sourcePath)
$regions=[ordered]@{
    history0=@(1020,280,280,210)
    history1=@(1000,270,320,240)
    history2=@(970,250,360,270)
    current=@(940,210,480,360)
    template=@(1100,310,105,105)
    search0=@(990,200,320,320)
    search1=@(1020,230,260,260)
    search2=@(965,210,320,320)
    search3=@(1020,210,300,300)
    candidate0=@(1050,280,210,210)
    candidate1=@(1080,300,170,170)
    candidate2=@(1020,260,240,240)
    candidate3=@(1060,270,210,210)
    output=@(1030,280,240,180)
}
$items=[ordered]@{}
$codec=[Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object {$_.MimeType -eq 'image/jpeg'}
$parameters=New-Object Drawing.Imaging.EncoderParameters 1
$parameters.Param[0]=New-Object Drawing.Imaging.EncoderParameter ([Drawing.Imaging.Encoder]::Quality),([long]94)
foreach($name in $regions.Keys){
    $r=$regions[$name]
    $bitmap=New-Object Drawing.Bitmap ([int]$r[2]),([int]$r[3])
    $g=[Drawing.Graphics]::FromImage($bitmap)
    $g.DrawImage($source,(New-Object Drawing.Rectangle 0,0,$r[2],$r[3]),$r[0],$r[1],$r[2],$r[3],[Drawing.GraphicsUnit]::Pixel)
    $path=Join-Path $AssetDirectory ($name+'.jpg')
    $bitmap.Save($path,$codec,$parameters)
    $g.Dispose();$bitmap.Dispose()
    $items[$name]=[ordered]@{file=$name+'.jpg';crop=$r;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$metadata=[ordered]@{
    source_url='https://source.roboflow.com/WD8jyT4rYRekug52HV2T/01QmL7sdP4j5CPoNxUiB/original.jpg'
    source_page='https://universe.roboflow.com/lab-lpaya/visdrone-ocpsh'
    attribution='visdrone Dataset, lab, Roboflow Universe (2022); source page lists CC BY 4.0.'
    source_sha256=(Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    source_size=@($source.Width,$source.Height)
    target_box_original=@(1124,337,58,49)
    note='All patches are illustrative crops of one real image. Boxes are manually placed for the architecture diagram, not GT, tracker outputs, consecutive frames, or experimental results.'
    assets=$items
}
$json=$metadata | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $AssetDirectory 'provenance.json'),$json,(New-Object Text.UTF8Encoding $false))
$parameters.Dispose();$source.Dispose()
Write-Output ('Prepared '+$items.Count+' photographic patches.')
