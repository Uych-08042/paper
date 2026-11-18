param([Parameter(Mandatory=$true)][string]$SvgPath, [int]$Scale=2)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
[xml]$svg=Get-Content -LiteralPath $SvgPath -Raw -Encoding UTF8
$inv=[Globalization.CultureInfo]::InvariantCulture
function Num($value){return [double]::Parse([string]$value,$inv)}
function Col($value){return [Drawing.ColorTranslator]::FromHtml([string]$value)}
$width=[int](Num $svg.DocumentElement.GetAttribute('width'))
$height=[int](Num $svg.DocumentElement.GetAttribute('height'))
$bitmap=New-Object Drawing.Bitmap ($width*$Scale),($height*$Scale)
$bitmap.SetResolution(300,300)
$g=[Drawing.Graphics]::FromImage($bitmap)
$g.Clear([Drawing.Color]::White)
$g.ScaleTransform($Scale,$Scale)
$g.SmoothingMode=[Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint=[Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$format=[Drawing.StringFormat]::GenericTypographic.Clone()
$format.FormatFlags=$format.FormatFlags -bor [Drawing.StringFormatFlags]::MeasureTrailingSpaces
foreach($node in $svg.DocumentElement.ChildNodes){
    $tag=$node.LocalName
    if($tag -in @('defs','title','desc')){continue}
    $fill=$node.GetAttribute('fill');$stroke=$node.GetAttribute('stroke')
    $pen=$null;$brush=$null
    if($fill -and $fill -ne 'none'){$brush=New-Object Drawing.SolidBrush (Col $fill)}
    if($stroke -and $stroke -ne 'none'){
        $sw=if($node.HasAttribute('stroke-width')){Num $node.GetAttribute('stroke-width')}else{1.0}
        $pen=New-Object Drawing.Pen (Col $stroke),([single]$sw)
        $pen.LineJoin=[Drawing.Drawing2D.LineJoin]::Round
        if($node.HasAttribute('stroke-dasharray')){$pen.DashPattern=[single[]]@(3.5,2.5)}
    }
    switch($tag){
        'rect'{
            $x=Num $node.GetAttribute('x');$y=Num $node.GetAttribute('y')
            $w=Num $node.GetAttribute('width');$h=Num $node.GetAttribute('height')
            $r=if($node.HasAttribute('rx')){Num $node.GetAttribute('rx')}else{0}
            $path=New-Object Drawing.Drawing2D.GraphicsPath
            if($r -gt 0){
                $d=[single]($r*2)
                $path.AddArc([single]$x,[single]$y,$d,$d,180,90)
                $path.AddArc([single]($x+$w-$d),[single]$y,$d,$d,270,90)
                $path.AddArc([single]($x+$w-$d),[single]($y+$h-$d),$d,$d,0,90)
                $path.AddArc([single]$x,[single]($y+$h-$d),$d,$d,90,90)
                $path.CloseFigure()
            }else{$path.AddRectangle((New-Object Drawing.RectangleF ([single]$x),([single]$y),([single]$w),([single]$h)))}
            if($brush){$g.FillPath($brush,$path)}
            if($pen){$g.DrawPath($pen,$path)}
            $path.Dispose()
        }
        'circle'{
            $x=Num $node.GetAttribute('cx');$y=Num $node.GetAttribute('cy');$r=Num $node.GetAttribute('r')
            if($brush){$g.FillEllipse($brush,[single]($x-$r),[single]($y-$r),[single](2*$r),[single](2*$r))}
        }
        'path'{
            $values=[regex]::Matches($node.GetAttribute('d'),'-?\d+(?:\.\d+)?')
            $points=New-Object 'Collections.Generic.List[Drawing.PointF]'
            for($i=0;$i -lt $values.Count;$i+=2){$points.Add((New-Object Drawing.PointF ([single](Num $values[$i].Value)),([single](Num $values[$i+1].Value))))}
            if($pen -and $points.Count -gt 1){
                $g.DrawLines($pen,$points.ToArray())
                if($node.HasAttribute('marker-end')){
                    $a=$points[$points.Count-2];$b=$points[$points.Count-1]
                    $angle=[math]::Atan2($b.Y-$a.Y,$b.X-$a.X)
                    $len=6*$pen.Width;$half=2.9*$pen.Width
                    $bx=$b.X-$len*[math]::Cos($angle);$by=$b.Y-$len*[math]::Sin($angle)
                    $arrow=[Drawing.PointF[]]@($b,
                        (New-Object Drawing.PointF ([single]($bx+$half*[math]::Sin($angle))),([single]($by-$half*[math]::Cos($angle)))),
                        (New-Object Drawing.PointF ([single]($bx-$half*[math]::Sin($angle))),([single]($by+$half*[math]::Cos($angle)))))
                    $arrowBrush=New-Object Drawing.SolidBrush $pen.Color
                    $g.FillPolygon($arrowBrush,$arrow);$arrowBrush.Dispose()
                }
            }
        }
        'text'{
            $size=Num $node.GetAttribute('font-size');$x=Num $node.GetAttribute('x');$y=Num $node.GetAttribute('y')
            $family=if($node.GetAttribute('font-family').Contains('Microsoft YaHei')){'Microsoft YaHei'}else{'Arial'}
            $style=if([int]$node.GetAttribute('font-weight') -ge 600){[Drawing.FontStyle]::Bold}else{[Drawing.FontStyle]::Regular}
            $font=New-Object Drawing.Font $family,([single]$size),$style,([Drawing.GraphicsUnit]::Pixel)
            $measure=$g.MeasureString($node.InnerText,$font,10000,$format)
            switch($node.GetAttribute('text-anchor')){'middle'{$x-=$measure.Width/2}'end'{$x-=$measure.Width}}
            $ascent=$font.FontFamily.GetCellAscent($style)*$size/$font.FontFamily.GetEmHeight($style)
            if($brush){$g.DrawString($node.InnerText,$font,$brush,[single]$x,[single]($y-$ascent),$format)}
            $font.Dispose()
        }
        default{throw "Unsupported SVG primitive: $tag"}
    }
    if($pen){$pen.Dispose()};if($brush){$brush.Dispose()}
}
$png=[IO.Path]::ChangeExtension($SvgPath,'.png')
$bitmap.Save($png,[Drawing.Imaging.ImageFormat]::Png)
$format.Dispose();$g.Dispose();$bitmap.Dispose()
Write-Output $png
