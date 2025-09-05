# Aircraft Analytics 001

This project simulates aircraft telemetry data and provides descriptive statistics for key metrics using Python.

## Files
- `aircraft_analytics_001.py`: Main script for simulation and analytics
- `requirements.txt`: Python dependencies

## Requirements
- Python 3.8+
- pandas
- numpy
## Virtual Environment (Recommended)
It is recommended to create a virtual environment before installing dependencies:

Sample

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
## Installation
1. Open a terminal in this directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the script to generate simulated data and view analytics:
```
python aircraft_analytics_001.py
```

## Output
- Simulated ADS-B snapshot (DataFrame)
- Descriptive statistics for:
  - Altitude (ft)
  - Ground Speed (kts)
  - Vertical Rate (fpm)
  - Position Confidence

## License
This project is for educational and analytical purposes.
