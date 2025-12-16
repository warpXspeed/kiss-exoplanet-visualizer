# KISS Exoplanet Visualizer

Then open your browser to http://localhost:8501 and pick any system from the sidebar.
What You NeedPython 3.8 or newer
The files in this repo:exo_visualizer_orbital.py→ the app
filled_systems.csv → the data
requirements.txt → the libraries it needs











Simple Streamlit app visualizing exoplanet orbital architectures from Kepler data. Displays known planets (cyan) and geometrically predicted "unseen" planets (magenta) to fill gaps, revealing natural harmony in spacing.

Inspired by KISS (Keep It Simple, Stupid) principle: Elegant geometric fills create balanced systems, just like nature.

## Features
- **1D Linear View**: Log-scaled scatter with star, dashed connections, and habitable zone (HZ) band.
- **2D Orbital Overview**: Top-down polar plot with concentric orbits, random planet positions, and sqrt scaling for inner visibility.
- Dark mode, HZ toggle, system selector.
- Highlights: KOI-351 (resonant chain), TRAPPIST-1 (compact), 55 Cnc (wide).
# KISS Exoplanet Visualizer 🌌

A simple Streamlit app that visualizes real exoplanet systems from Kepler data, plus my own geometric predictions for possible missing planets.

- **Cyan dots** = confirmed known planets  
- **Magenta stars** = my predicted "unseen" planets  
- Two views: classic 1D log-scale diagram + beautiful 2D top-down orbital overview  
- Dark mode, habitable zone toggle, and easy system selector

Best-looking systems: KOI-351, TRAPPIST-1, 55 Cnc

## How to Run It

```bash
git clone https://github.com/warpXspeed/kiss-exoplanet-visualizer.git
cd kiss-exoplanet-visualizer
pip install -r requirements.txt
streamlit run exo_visualizer_orbital.py

Open in browser (localhost:8501). Select a system from the sidebar.

## Data
- `filled_systems.csv`: Pre-processed Kepler data with systems, stellar masses, known planets (period days + semi-major axis AU), and predicted unseens.
- Format example for planets: "b 1.5d (0.011 AU); c 3.2d (0.021 AU)"
- Update CSV to add more systems or refine predictions.

## Dependencies
- Python 3.8+
- See `requirements.txt` for Streamlit, Pandas, Matplotlib, NumPy.

## Customization
- Add eccentricity/inclination: Extend `parse_planets` to handle " (a AU, e=0.1, i=89°)" and draw ellipses in 2D.
- Deploy to Streamlit Sharing: Push to GitHub, sign up at share.streamlit.io, deploy from repo.
- HZ Calculation: Rough estimate based on stellar mass (L ~ M^3.5); tweak bounds as needed.

## Contributing
Fork and PR! Ideas: Animations, eccentricity support, more data integration.

## License
MIT

Built with ❤️ for exoplanet enthusiasts. Share on X/r/exoplanets!



