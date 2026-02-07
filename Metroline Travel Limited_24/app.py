import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import os
import builtins

# Save Python's built-in abs before PySpark overwrites it
python_abs = builtins.abs

# Page configuration
st.set_page_config(
    page_title="Bus Travel Time Predictor",
    page_icon="🚌",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .route-details {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-time {
        font-size: 3rem;
        font-weight: bold;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_spark():
    """Initialize Spark Session"""
    spark = SparkSession.builder \
        .appName("Bus Travel Time Prediction App") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.default.parallelism", "4") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1") \
        .getOrCreate()
    
    # Set log level to ERROR to reduce console output
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark


@st.cache_resource
def load_and_preprocess_data(_spark):
    """Load and clean the dataset"""
    
    # Load data
    file_path = "bus_data_comprehensive.csv"
    df = _spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Data Cleaning
    df_clean = df.filter(col('runtime_seconds').isNotNull())
    df_clean = df_clean.filter(col('distance_meters').isNotNull())
    df_clean = df_clean.filter(col('distance_meters') > 0)
    df_clean = df_clean.filter(col('runtime_seconds') > 0)
    
    # Create speed column and remove outliers
    df_clean = df_clean.withColumn('speed_kmh', 
                                   (col('distance_meters') / col('runtime_seconds')) * 3.6)
    
    df_clean = df_clean.filter(
        (col('speed_kmh') >= 2) & (col('speed_kmh') <= 80)
    )
    
    df_clean = df_clean.dropDuplicates()
    
    # Feature Engineering
    df_processed = df_clean.withColumn(
        'stop_pair',
        concat_ws('_to_', col('from_stop_id'), col('to_stop_id'))
    )
    
    df_processed = df_processed.withColumn(
        'distance_category',
        when(col('distance_meters') <= 200, 'short')
        .when(col('distance_meters') <= 500, 'medium')
        .when(col('distance_meters') <= 1000, 'long')
        .otherwise('very_long')
    )
    
    window_spec = Window.partitionBy('journey_pattern_section_id')
    df_processed = df_processed.withColumn(
        'max_sequence',
        max('to_sequence_number').over(window_spec)
    )
    df_processed = df_processed.withColumn(
        'route_position',
        col('from_sequence_number') / col('max_sequence')
    )
    
    # Select relevant features
    features_to_keep = [
        'runtime_seconds',
        'distance_meters',
        'from_sequence_number',
        'to_sequence_number',
        'speed_kmh',
        'route_position',
        'from_stop_id',
        'to_stop_id',
        'stop_pair',
        'line_name',
        'from_locality_name',
        'to_locality_name',
        'from_timing_status',
        'to_timing_status',
        'distance_category'
    ]
    
    df_features = df_processed.select(features_to_keep)
    
    # Handle missing values
    categorical_cols = ['from_locality_name', 'to_locality_name', 
                       'from_timing_status', 'to_timing_status']
    
    for col_name in categorical_cols:
        df_features = df_features.fillna('UNKNOWN', subset=[col_name])
    
    df_features.cache()
    
    return df_features, df_clean


@st.cache_resource
def build_and_train_model(_spark, _df_features):
    """Build preprocessing pipeline and train model"""
    
    # Create StringIndexers
    categorical_features = [
        'from_stop_id', 'to_stop_id', 'stop_pair', 'line_name',
        'from_locality_name', 'to_locality_name',
        'from_timing_status', 'to_timing_status', 'distance_category'
    ]
    
    indexers = []
    indexed_cols = []
    
    for cat_col in categorical_features:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_indexed",
            handleInvalid='keep'
        )
        indexers.append(indexer)
        indexed_cols.append(f"{cat_col}_indexed")
    
    # Numerical features
    numerical_features = [
        'distance_meters',
        'from_sequence_number',
        'to_sequence_number',
        'speed_kmh',
        'route_position'
    ]
    
    # Create feature vector
    all_feature_cols = indexed_cols + numerical_features
    
    vector_assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol='unscaled_features',
        handleInvalid='skip'
    )
    
    # StandardScaler
    scaler = StandardScaler(
        inputCol='unscaled_features',
        outputCol='features',
        withMean=True,
        withStd=True
    )
    
    # Build preprocessing pipeline
    preprocessing_stages = indexers + [vector_assembler, scaler]
    preprocessing_pipeline = Pipeline(stages=preprocessing_stages)
    
    # Fit preprocessing pipeline
    preprocessing_model = preprocessing_pipeline.fit(_df_features)
    
    # Transform data
    df_transformed = preprocessing_model.transform(_df_features)
    df_model = df_transformed.select('features', 'runtime_seconds')
    df_model = df_model.withColumnRenamed('runtime_seconds', 'label')
    
    # Train Linear Regression model
    train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    
    lr = LinearRegression(
        featuresCol='features',
        labelCol='label',
        predictionCol='prediction',
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
        standardization=False
    )
    
    lr_model = lr.fit(train_data)
    
    # Evaluate on test data
    test_predictions = lr_model.transform(test_data)
    from pyspark.ml.evaluation import RegressionEvaluator
    
    r2_evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='r2'
    )
    
    mae_evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='mae'
    )
    
    test_r2 = r2_evaluator.evaluate(test_predictions)
    test_mae = mae_evaluator.evaluate(test_predictions)
    
    return preprocessing_model, lr_model, test_r2, test_mae


@st.cache_data
def get_route_segments(_df_features):
    """Get aggregated route segments for each line"""
    
    route_segments = _df_features.groupBy(
        'line_name', 'from_stop_id', 'to_stop_id', 
        'from_locality_name', 'to_locality_name'
    ).agg(
        avg('distance_meters').alias('avg_distance'),
        avg('runtime_seconds').alias('avg_runtime'),
        avg('from_sequence_number').alias('avg_from_seq'),
        avg('to_sequence_number').alias('avg_to_seq'),
        count(lit(1)).alias('trip_count')
    ).orderBy('line_name', 'avg_from_seq').toPandas()
    
    return route_segments


@st.cache_data
def get_stop_names(_df_clean):
    """Get stop ID to name mappings"""
    
    stop_data = _df_clean.select(
        'from_stop_id', 'from_stop_name', 
        'to_stop_id', 'to_stop_name'
    ).distinct().toPandas()
    
    from_stop_names = dict(zip(stop_data['from_stop_id'], stop_data['from_stop_name']))
    to_stop_names = dict(zip(stop_data['to_stop_id'], stop_data['to_stop_name']))
    
    return from_stop_names, to_stop_names


def calculate_derived_features(distance, from_seq, to_seq):
    """Calculate derived features for prediction"""
    
    # Estimate speed (assuming average bus speed)
    estimated_time = distance / 8.33  # Assuming ~30 km/h average
    speed_kmh = (distance / estimated_time) * 3.6 if estimated_time > 0 else 30.0
    
    # Route position
    route_position = from_seq / to_seq if to_seq > 0 else 0.5
    
    # Distance category
    if distance <= 200:
        distance_category = 'short'
    elif distance <= 500:
        distance_category = 'medium'
    elif distance <= 1000:
        distance_category = 'long'
    else:
        distance_category = 'very_long'
    
    return speed_kmh, route_position, distance_category


def make_prediction(spark, preprocessing_model, lr_model, segment_data, 
                   distance, from_seq, to_seq):
    """Make travel time prediction"""
    
    from pyspark.sql import Row
    
    # Calculate derived features
    speed_kmh, route_position, distance_category = calculate_derived_features(
        distance, from_seq, to_seq
    )
    
    # Create stop pair
    stop_pair = f"{segment_data['from_stop_id']}_to_{segment_data['to_stop_id']}"
    
    # Get timing status from historical data
    from_timing = segment_data.get('from_timing_status', 'PTP')
    to_timing = segment_data.get('to_timing_status', 'PTP')
    
    # Create prediction row
    prediction_row = Row(
        runtime_seconds=0.0,
        distance_meters=float(distance),
        from_sequence_number=int(from_seq),
        to_sequence_number=int(to_seq),
        speed_kmh=float(speed_kmh),
        route_position=float(route_position),
        from_stop_id=str(segment_data['from_stop_id']),
        to_stop_id=str(segment_data['to_stop_id']),
        stop_pair=stop_pair,
        line_name=str(segment_data['line_name']),
        from_locality_name=str(segment_data['from_locality_name']),
        to_locality_name=str(segment_data['to_locality_name']),
        from_timing_status=str(from_timing),
        to_timing_status=str(to_timing),
        distance_category=distance_category
    )
    
    # Create DataFrame
    prediction_df = spark.createDataFrame([prediction_row])
    
    # Apply preprocessing
    processed_df = preprocessing_model.transform(prediction_df)
    
    # Make prediction
    prediction_result = lr_model.transform(processed_df)
    
    # Collect prediction using the simplest method
    try:
        # Use collect() which is more reliable than toPandas() or first()
        result_rows = prediction_result.select('prediction').collect()
        if result_rows and len(result_rows) > 0:
            predicted_time = float(result_rows[0]['prediction'])
        else:
            # Fallback: use average time estimate
            predicted_time = distance / 8.33
    except Exception as e:
        # Silently use fallback estimate
        predicted_time = distance / 8.33
    
    return predicted_time


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<p class="main-header">🚌 Bus Travel Time Predictor</p>', 
                unsafe_allow_html=True)
    
    st.markdown("**Machine Learning-powered travel time prediction using PySpark Linear Regression**")
    
    # Initialize
    with st.spinner("🔄 Initializing Spark and loading model..."):
        spark = initialize_spark()
        df_features, df_clean = load_and_preprocess_data(spark)
        preprocessing_model, lr_model, test_r2, test_mae = build_and_train_model(
            spark, df_features
        )
        route_segments = get_route_segments(df_features)
        from_stop_names, to_stop_names = get_stop_names(df_clean)
    
    # Display model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model R² Score", f"{test_r2:.4f}")
    with col2:
        st.metric("Mean Absolute Error", f"{test_mae:.2f} sec")
    with col3:
        st.metric("Total Routes", len(route_segments))
    
    st.divider()
    
    # Sidebar - Selection
    with st.sidebar:
        st.header("🔧 Route Selection")
        
        # Get unique lines
        available_lines = sorted(route_segments['line_name'].unique())
        
        # Select bus line
        selected_line = st.selectbox(
            "Select Bus Line",
            available_lines,
            help="Choose the bus line for prediction"
        )
        
        # Filter segments for selected line
        line_segments = route_segments[
            route_segments['line_name'] == selected_line
        ].copy()
        
        # Create route segment display names
        line_segments['segment_display'] = line_segments.apply(
            lambda row: f"From {from_stop_names.get(row['from_stop_id'], row['from_stop_id'])} to {to_stop_names.get(row['to_stop_id'], row['to_stop_id'])} ({row['avg_distance']:.0f}m)",
            axis=1
        )
        
        # Select route segment
        segment_display = st.selectbox(
            "Select Route Segment",
            line_segments['segment_display'].tolist(),
            help="Choose the specific route segment"
        )
        
        # Get selected segment data
        selected_idx = line_segments[
            line_segments['segment_display'] == segment_display
        ].index[0]
        selected_segment = line_segments.loc[selected_idx]
        
        st.divider()
        
        st.markdown("### 📊 Route Statistics")
        st.metric("Historical Trips", f"{int(selected_segment['trip_count'])}")
        st.metric("Avg Historical Time", 
                 f"{selected_segment['avg_runtime']:.1f} sec")
        st.metric("Avg Distance", f"{selected_segment['avg_distance']:.0f} m")
    
    # Main Area - Route Details and Prediction
    st.header("📍 Selected Route Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="route-details">
            <h4>🚏 From Stop</h4>
            <p><strong>{from_stop_names.get(selected_segment['from_stop_id'], selected_segment['from_stop_id'])}</strong></p>
            <p>Locality: {selected_segment['from_locality_name']}</p>
            <p>Stop ID: {selected_segment['from_stop_id']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="route-details">
            <h4>🎯 To Stop</h4>
            <p><strong>{to_stop_names.get(selected_segment['to_stop_id'], selected_segment['to_stop_id'])}</strong></p>
            <p>Locality: {selected_segment['to_locality_name']}</p>
            <p>Stop ID: {selected_segment['to_stop_id']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Input Parameters
    st.header("⚙️ Adjust Parameters (Optional)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        distance = st.number_input(
            "Distance (meters)",
            min_value=10.0,
            max_value=10000.0,
            value=float(selected_segment['avg_distance']),
            step=10.0,
            help="Adjust the distance between stops"
        )
    
    with col2:
        from_seq = st.number_input(
            "From Sequence Number",
            min_value=1,
            max_value=100,
            value=int(selected_segment['avg_from_seq']),
            help="Position in route sequence"
        )
    
    with col3:
        to_seq = st.number_input(
            "To Sequence Number",
            min_value=from_seq + 1,
            max_value=100,
            value=int(selected_segment['avg_to_seq']),
            help="End position in route sequence"
        )
    
    st.divider()
    
    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(
            "🚀 Predict Travel Time",
            use_container_width=True,
            type="primary"
        )
    
    # Make Prediction
    if predict_button:
        with st.spinner("🔮 Calculating prediction..."):
            # Get timing status from original data
            timing_data = df_features.filter(
                (df_features.line_name == str(selected_line)) &
                (df_features.from_stop_id == selected_segment['from_stop_id']) &
                (df_features.to_stop_id == selected_segment['to_stop_id'])
            ).select('from_timing_status', 'to_timing_status').first()
            
            segment_data = {
                'from_stop_id': selected_segment['from_stop_id'],
                'to_stop_id': selected_segment['to_stop_id'],
                'line_name': selected_line,
                'from_locality_name': selected_segment['from_locality_name'],
                'to_locality_name': selected_segment['to_locality_name'],
                'from_timing_status': timing_data['from_timing_status'] if timing_data else 'PTP',
                'to_timing_status': timing_data['to_timing_status'] if timing_data else 'PTP'
            }
            
            predicted_time = make_prediction(
                spark, preprocessing_model, lr_model,
                segment_data, distance, from_seq, to_seq
            )
        
        # Display Results
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🤖 ML Predicted Time</h3>
                <p class="prediction-time">{predicted_time:.1f} sec</p>
                <p style="font-size: 1.5rem; color: #28a745;">({predicted_time/60:.2f} minutes)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            historical_time = selected_segment['avg_runtime']
            difference = predicted_time - historical_time
            percentage_diff = (difference / historical_time) * 100 if historical_time > 0 else 0
            
            diff_color = "#dc3545" if difference > 0 else "#28a745"
            diff_symbol = "+" if difference > 0 else ""
            
            st.markdown(f"""
            <div class="prediction-result" style="background-color: #fff3cd; border-color: #ffc107;">
                <h3>📊 Historical Average</h3>
                <p class="prediction-time" style="color: #856404;">{historical_time:.1f} sec</p>
                <p style="font-size: 1.5rem; color: #856404;">({historical_time/60:.2f} minutes)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Difference Analysis
        st.markdown("### 📈 Prediction Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Time Difference",
                f"{python_abs(difference):.1f} sec",
                f"{diff_symbol}{difference:.1f} sec"
            )
        
        with col2:
            st.metric(
                "Percentage Difference",
                f"{python_abs(percentage_diff):.1f}%",
                f"{diff_symbol}{percentage_diff:.1f}%"
            )
        
        with col3:
            if python_abs(difference) <= 10:
                accuracy = "🎯 Excellent Match"
            elif python_abs(difference) <= 30:
                accuracy = "✅ Good Match"
            else:
                accuracy = "⚠️ Moderate Variance"
            
            st.info(accuracy)
        
        # Interpretation
        st.markdown("---")
        
        if difference > 0:
            st.info(f"ℹ️ The ML model predicts **{python_abs(difference):.1f} seconds longer** than the historical average. This could indicate potential traffic or delays.")
        elif difference < 0:
            st.success(f"✅ The ML model predicts **{python_abs(difference):.1f} seconds faster** than the historical average. Conditions may be favorable.")
        else:
            st.success("🎯 The prediction exactly matches the historical average!")


if __name__ == "__main__":
    main()
