# Bus Travel Time Predictor - Streamlit App

A machine learning-powered web application for predicting bus travel times using PySpark and Streamlit.

## Features

- 🚌 **Interactive Route Selection**: Choose bus lines and specific route segments
- 🤖 **ML-Powered Predictions**: Uses Linear Regression trained on historical data
- 📊 **Real-time Analysis**: Compare predictions with historical averages
- ⚙️ **Customizable Parameters**: Adjust distance and sequence numbers for "what-if" scenarios
- 📈 **Performance Metrics**: View model accuracy and prediction confidence

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

1. Make sure `bus_data_comprehensive.csv` is in the same directory as `app.py`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The app will open in your default browser (usually at http://localhost:8501)

## How to Use

### Sidebar Controls:
1. **Select Bus Line**: Choose from available bus lines
2. **Select Route Segment**: Pick a specific route between stops
3. View **Route Statistics**: Historical trips, average time, and distance

### Main Interface:
1. **Route Details**: See departure and arrival stop information
2. **Adjust Parameters**: Optionally modify distance or sequence numbers
3. **Predict**: Click "Predict Travel Time" button
4. **Results**: View ML prediction vs historical average with analysis

## Model Details

- **Algorithm**: Linear Regression (PySpark ML)
- **Features**: 14 features including distance, sequence, locality, timing status
- **Preprocessing**: StringIndexing, VectorAssembly, StandardScaling
- **Caching**: Model and data are cached for fast predictions

## Technical Architecture

- **Frontend**: Streamlit
- **Backend**: PySpark (Distributed ML)
- **Data Processing**: Real-time feature engineering
- **Model**: Cached using @st.cache_resource for performance

## Performance

The model is trained once when the app starts (cached), then provides instant predictions for any route configuration.

---

Built with ❤️ using PySpark and Streamlit
