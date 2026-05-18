#!/bin/bash
# Stage 02: SurtGis terrain + hydrology over FABDEM tile.
set -e
cd "${PIPELINE_WORKDIR:-$(dirname "$0")/..}"
echo "[02_terrain] WORKDIR=$(pwd)"

DEM="dem/fabdem.tif"
[ ! -f "$DEM" ] && echo "Missing $DEM" && exit 1

if [ -f "factors/slope.tif" ] && [ -f "hydro/hand.tif" ] && [ -f "factors/valley_depth.tif" ]; then
    echo "✓ terrain+hydrology already done"
    exit 0
fi

mkdir -p factors hydro

echo "=== surtgis terrain all ==="
time surtgis terrain all "$DEM" --outdir factors/ --compress

echo ""
echo "=== surtgis hydrology all ==="
time surtgis hydrology all "$DEM" --outdir hydro/ --compress

echo ""
echo "=== valley-depth (single) ==="
surtgis terrain valley-depth "$DEM" --compress factors/valley_depth.tif

echo "Done."
