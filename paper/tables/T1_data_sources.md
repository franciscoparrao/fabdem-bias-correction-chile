# T1 — Data sources

| Source            | Type                                           | Native res.                   | Coverage                                  | Format        | License           | Access                                            |
|:------------------|:-----------------------------------------------|:------------------------------|:------------------------------------------|:--------------|:------------------|:--------------------------------------------------|
| FABDEM v1.2       | Bare-earth DEM                                 | 1 arcsec ($\sim$30 m)         | $\pm$80$^{\circ}$ lat                     | GeoTIFF       | CC BY-NC-4.0      | GEE \texttt{projects/sat-io/open-datasets/FABDEM} |
| ICESat-2 ATL08 v7 | Photon-derived land + canopy heights           | 100~m segment, 17~m footprint | Global, 91-day repeat                     | HDF5          | Public domain     | NASA Earthdata via \texttt{earthaccess}           |
| Sentinel-2 L2A    | Surface reflectance, 13 bands                  | 10--60 m                      | Global, $\sim$5-day revisit               | GeoTIFF (COG) | Open (Copernicus) | Microsoft Planetary Computer STAC                 |
| Sentinel-1 RTC    | C-band SAR backscatter $\sigma^{0}$            | 10~m                          | Global, $\sim$12-day revisit              | GeoTIFF (COG) | Open (Copernicus) | Microsoft Planetary Computer STAC                 |
| EGM2008 geoid     | Geoid undulation grid                          | 2.5$^{\prime}$                | Global                                    | GeoTIFF       | Open (NGA)        | PROJ network mode (EPSG:4979$\rightarrow$9518)    |
| CIGIDEN Licantén  | Sentinel-derived flood polygon, jun 2023 event | Vector polygons               | Licantén (35$^{\circ}$S), Mataquito mouth | Shapefile     | Open (CC)         | Zenodo DOI \texttt{10.5281/zenodo.13307972}       |
