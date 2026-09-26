# Baseline table assembly

`assemble.py` joins the released dynamic-Gabor, static-Gabor, MESI, natural, and drifting-grating response tables by exact `(session, scan, readout)` keys. It never falls back to an older cohort and never imputes missing pairs. Stimulus-specific generation interfaces live in `src/mesi/`, `src/gabor/`, `src/grating/`, and `src/natural/`.
