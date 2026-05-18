#!/bin/bash
# E1a stage 02: SurtGis terrain + hydrology over FABDEM tile.
set -e
cd "$(dirname "$0")/.."

DEM="dem/fabdem.tif"
[ ! -f "$DEM" ] && echo "Missing $DEM" && exit 1

if [ -f "factors/slope.tif" ] && [ -f "hydro/hand.tif" ]; then
    echo "✓ terrain+hydrology already done"
    exit 0
fi

echo "=== surtgis terrain all ==="
time surtgis terrain all "$DEM" --outdir factors/ --compress

echo ""
echo "=== surtgis hydrology all ==="
time surtgis hydrology all "$DEM" --outdir hydro/ --compress

echo ""
echo "=== valley-depth (single) ==="
surtgis terrain valley-depth "$DEM" --compress factors/valley_depth.tif

echo ""
echo "Done. Outputs:"
ls -lh factors/ hydro/ | tail -40
