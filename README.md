# Forex Trading Bot

A beginner-friendly Python project for downloading and analyzing forex data.

## Project Structure

```
forex-bot/
├── venv/              # Virtual environment (dependencies)
├── src/               # Source code
│   └── main.py        # Main script
├── data/              # Data output directory
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Setup Instructions

### 1. Activate Virtual Environment

On Windows (Git Bash):

```bash
source venv/Scripts/activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

### 2. Install Dependencies

If you haven't already, install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Script

Once the virtual environment is activated, run:

```bash
python src/main.py
```

This will:

- Download 1 year of EURUSD exchange rate data
- Display summary statistics
- Create a plot and save it to `data/eurusd_plot.png`
- Display the plot on screen

## What's Being Used

- **yfinance**: Downloads forex and market data
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization

## Next Steps

You can extend this project by:

- Adding more currency pairs
- Calculating moving averages
- Adding technical indicators
- Creating a backtesting system
- Implementing trading signals
