# How far can open data take a small water utility? — scripts, models and results

Code and data for the paper *"How far can open data take a small water utility? Testing hydraulic models built from public data under continuous and intermittent supply"* by Leslie Allan Marshall (Allan Engineering Software Development, ORCID [0009-0009-9218-5674](https://orcid.org/0009-0009-9218-5674)), submitted to *Water Supply* (IWA Publishing). [Add journal DOI on publication.]

The study tests, on one benchmark and three real networks, how accurate a hydraulic model built from guessed or public data is under continuous supply and under four hours of supply per day:

- **Test A**: guess one input at a time (roughness, pipe sizes, ground level, demand spread) on the true layout.
- **Test B**: rebuild the network from OpenStreetMap streets and buildings and a 30 m elevation model.
- **Test C**: tune roughness to 1–8 simulated pressure loggers and predict under-supplied junctions.

## Contents

| Folder | What it holds |
|---|---|
| `core_protocol/` | `core.py` (Tests A and B), `stageC.py` (Test C), `figs/make_figs.py` (paper figures), `results/` (all result tables used in the paper) |
| `paper3_sensitivity/` | Supply model helpers (`run_iws.py`, `run_sensitivity.py`) and the L-Town preliminary and ground-level tests (`run_test1b.py`, results in `results/`) |
| `epanet/Networks/L-TOWN.inp` | L-Town benchmark network (BattLeDIM, CC BY 4.0) |
| `kitchener/` | Build scripts and EPANET models for Kitchener pressure zone KIT 6 (true, road-class rebuild, design-rule rebuild) |
| `christchurch/` | Build scripts and EPANET models for Christchurch pressure zone Mt Pleasant 3 |
| `lowerhutt/` | Build scripts only — see *Lower Hutt* below |

## Running

Python 3.10 with the packages in `requirements.txt` (`pip install -r requirements.txt`). From `core_protocol/`:

```
python core.py A                    # Test A, all networks available
python core.py B                    # Test B (Kitchener, Christchurch, and Lower Hutt if built)
python stageC.py global CHCH        # Test C, loggers read on a normal day
python stageC.py global4 CHCH       # Test C, loggers read during supply hours
python stageC.py zone CHCH all_guessed 0 1 2 3 4   # Test C, three districts
python figs/make_figs.py <outdir> fig2 fig3 fig4 fig6 fig7
```

Network codes: `LTOWN`, `KIT6`, `CHCH`, `LHUTT`. Results are written to `core_protocol/results/`; cached simulations (`_truth_*.pkl`, `_bank_*.pkl`) are created on first run and are not distributed. Runs use the EPANET 2.2 solver through WNTR 1.5. The CSV files in `results/` are the exact tables behind the paper.

## Lower Hutt

The Lower Hutt (Konini Parkway district metered area) network data are owned by **Tiaki Wai Limited** (data owner and custodian, formerly Wellington Water) and were used **with permission** for this research. The data and the models built from them are **not redistributed** here. Aggregated results for Lower Hutt are included in `core_protocol/results/` (files containing `LHUTT`). To reproduce the Lower Hutt models, obtain the three waters asset data from Tiaki Wai Limited (Tiaki Wai Open Data Portal) under its terms and run `lowerhutt/build_true_graph.py`, `lowerhutt/build_true_inp.py` and `lowerhutt/rebuild_compare.py`. Without them, `core.py` skips Lower Hutt automatically. Tiaki Wai states that its asset data are incomplete in places and not suitable for detailed engineering design; the Lower Hutt model is a research reconstruction, not Tiaki Wai's operational model.

## Data sources and licences

See `DATA_LICENCES.md`. Raw inputs (pipe layers, OpenStreetMap extracts, elevation tiles) are not included; the build scripts document how each model was made from them.

## Licence

Code: MIT (see `LICENSE`). Network models and results: the licence of the source data applies (see `DATA_LICENCES.md`).

## Citation

If you use this material, please cite the paper (and this repository's Zenodo DOI once issued). See `CITATION.cff`.
