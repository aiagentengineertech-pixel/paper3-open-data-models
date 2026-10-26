# Data sources and licences

| Data | Source | Licence / terms | Included here |
|---|---|---|---|
| L-Town benchmark network | Vrachimis et al. 2020, *Dataset of BattLeDIM*, Zenodo, https://doi.org/10.5281/zenodo.4017659 | CC BY 4.0 | `epanet/Networks/L-TOWN.inp` |
| Kitchener water mains (pressure zone KIT 6) | City of Kitchener open data | Contains information licensed under the Open Government Licence – The Corporation of the City of Kitchener (v1.0) | Derived EPANET models in `kitchener/models/` |
| Christchurch water supply (Mt Pleasant 3) | Christchurch City Council Water Supply Open Data, https://gis.ccc.govt.nz/server/rest/services/OpenData/WaterSupply/FeatureServer | CC BY 4.0 | Derived EPANET models in `christchurch/models/` |
| Lower Hutt three waters asset data (Konini Parkway) | Tiaki Wai Limited (data owner and custodian), Tiaki Wai Open Data Portal | Used with permission of the data owner (9 October 2026). Tiaki Wai Limited must be acknowledged as data owner. No warranty; not for detailed engineering design. | **Not included.** Aggregated results only. |
| Street network and building footprints | © OpenStreetMap contributors, https://www.openstreetmap.org/copyright | Open Database Licence (ODbL) 1.0 | Rebuild models (`*_osm_*.inp`) are derived databases released under the ODbL |
| Ground elevation | Copernicus GLO-30 DEM, https://doi.org/10.5270/ESA-c5d3d65 | Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved | Elevations sampled into the models only |

No data from any Caribbean water utility were used.
