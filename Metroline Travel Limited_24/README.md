# Bus Travel Time Prediction using PySpark

## 📊 Overview

This project implements a complete **Big Data Analytics** solution for predicting bus travel times using Apache Spark and PySpark MLlib. It includes:
1. **Data Extraction**: XML to CSV converter that extracts stop-to-stop travel data from TransXChange format
2. **Data Processing**: PySpark-based data ingestion, cleaning, and feature engineering
3. **Machine Learning**: Multiple regression models (Linear Regression, Random Forest, Gradient Boosted Trees) for travel time prediction
4. **Evaluation & Visualization**: Comprehensive model evaluation with performance metrics and visualizations

## ✅ What Was Extracted

### Extraction Results
- **Files Processed**: 7 XML files
- **Total Records**: 40,342 stop-to-stop segments
- **Bus Lines**: 7 unique lines (107, 292, 317, 324, 331, 491, 81)
- **Unique Stops**: 532 stops across the network
- **Operator**: Metroline Travel Limited

### Data Completeness (ML-Ready)
| Field Category | Completeness | Status |
|----------------|--------------|--------|
| Stop IDs & Names | 100% | ✓✓✓ |
| Travel Time (Runtime) | 97.5% | ✓✓✓ |
| Distance | 100% | ✓✓✓ |
| Sequence Numbers | 100% | ✓✓✓ |
| Route/Line Info | 100% | ✓✓✓ |
| GPS Coordinates | 0% | ✗ (not in XML) |

**ML Readiness Score**: 99.4% - EXCELLENT for travel time prediction!

## 📁 Project Structure

```
Metroline Travel Limited_24/
│
├── Analysis.ipynb                          # Main PySpark ML analysis notebook
├── bus_data_comprehensive.csv              # Processed dataset (40,342 records)
├── xml_to_csv_comprehensive.py             # XML extraction script
├── xml_to_csv_flexible.py                  # Alternative extraction script
├── README.md                               # Project documentation
│
├── Metroline_TfL_9w6A35W/                  # Source XML data
│   ├── tfl_54-107-_-y05-57629.xml
│   ├── tfl_54-292-_-y05-61513.xml
│   ├── tfl_54-317-_-y05-55365.xml
│   ├── tfl_54-324-_-y05-52641.xml
│   ├── tfl_54-331-_-y05-60055.xml
│   ├── tfl_54-491-_-y05-55029.xml
│   └── tfl_54-81-_-y05-59103.xml
│
└── .ipynb_checkpoints/                     # Notebook autosaves
    └── Analysis-checkpoint.ipynb
```

## 📋 CSV Column Structure

### High Priority Columns (ML-Ready)
```
from_stop_id              - Origin stop identifier
from_stop_name            - Origin stop name
from_locality_name        - Origin locality
to_stop_id                - Destination stop identifier
to_stop_name              - Destination stop name
to_locality_name          - Destination locality
runtime_seconds           - Travel time in seconds (TARGET VARIABLE)
runtime_minutes           - Travel time in minutes
distance_meters           - Segment distance in meters
distance_km               - Segment distance in kilometers
```

### Medium Priority Columns
```
line_name                 - Bus line number/name
service_code              - Service identifier
operator_name             - Operator name
from_sequence_number      - Stop sequence on route
to_sequence_number        - Next stop sequence
route_link_ref            - Route link reference ID
from_stop_indicator       - Stop indicator (e.g., "Stop A")
to_stop_indicator         - Destination stop indicator
```

### Low Priority Columns (Metadata)
```
source_file               - Original XML filename
creation_date             - Data creation timestamp
modification_date         - Last modification timestamp
operator_code             - Operator code
operating_start_date      - Service start date
operating_end_date        - Service end date
from_activity             - Activity type (pickUp, setDown, etc.)
to_activity               - Activity type at destination
from_timing_status        - Timing point status
to_timing_status          - Timing point status
... (additional metadata fields)
```

## 📊 Data Statistics

### Travel Time Analysis
- **Mean Runtime**: 75 seconds (1.25 minutes)
- **Median Runtime**: 60 seconds (1.0 minutes)
- **Range**: 30 - 300 seconds

### Distance Analysis
- **Mean Distance**: 394 meters
- **Median Distance**: 346 meters
- **Range**: 17 - 2,064 meters

### Speed Analysis
- **Mean Speed**: 20.75 km/h (typical urban bus speed)
- **Median Speed**: 18.06 km/h
- **Range**: 0.51 - 178.70 km/h

## 🚀 Usage

### Step 1: Data Extraction (XML to CSV)
```bash
python xml_to_csv_comprehensive.py
```

This will:
1. Scan all XML files in the `Metroline_TfL_9w6A35W/` directory
2. Extract all available fields from TransXChange format
3. Create `bus_data_comprehensive.csv` with 40,342+ records
4. Print detailed summary and data quality report

### Step 2: Machine Learning Analysis (PySpark)
Open and run **Analysis.ipynb** in Jupyter Notebook:
```bash
jupyter notebook Analysis.ipynb
```

The notebook includes:
1. **Data Ingestion**: Load CSV into PySpark DataFrame
2. **Data Exploration**: Statistical analysis, missing values detection, schema validation
3. **Data Preprocessing**: Feature engineering, encoding categorical variables, handling nulls
4. **Feature Engineering**: VectorAssembler for ML pipeline, StandardScaler for normalization
5. **Model Training**: Train multiple regression models (Linear Regression, Random Forest, GBT)
6. **Model Evaluation**: Compare models using RMSE, MAE, R² metrics
7. **Visualization**: Performance charts, feature importance, prediction plots

### Alternative: Flexible Extraction
```bash
python xml_to_csv_flexible.py
```

For custom field selection and different XML structures.

## 🤖 Machine Learning Pipeline

### Problem Statement
**Objective**: Predict stop-to-stop travel time (`runtime_seconds`) for bus routes using PySpark MLlib

### Models Implemented
1. **Linear Regression** (Baseline)
   - Fast training and inference
   - Interpretable coefficients
   - Good for linear relationships

2. **Random Forest Regressor**
   - Handles non-linear patterns
   - Feature importance analysis
   - Robust to outliers

3. **Gradient Boosted Trees (GBT)**
   - State-of-the-art ensemble method
   - Sequential error correction
   - Best performance for complex patterns

### Feature Engineering
**Input Features**:
- `distance_meters` - Physical distance between stops
- `distance_km` - Distance in kilometers
- `from_stop_id`, `to_stop_id` - Stop identifiers (encoded)
- `line_name` - Bus line number
- `from_sequence_number`, `to_sequence_number` - Stop order
- `operator_name` - Transport operator
- Categorical encodings via StringIndexer and OneHotEncoder

**Target Variable**:
- `runtime_seconds` - Actual travel time (to predict)

### ML Pipeline Components
```python
1. StringIndexer → Convert categorical features to numeric indices
2. OneHotEncoder → Create binary vectors for categorical features
3. VectorAssembler → Combine features into single vector
4. StandardScaler → Normalize features for better convergence
5. ML Model → Train regression model
6. Pipeline → Chain all transformations
```

### Model Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Primary metric for prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error in seconds
- **R²** (R-squared): Proportion of variance explained
- **MSE** (Mean Squared Error): Squared error magnitude

### Key Insights from Analysis
- **Best Model**: Gradient Boosted Trees (lowest RMSE)
- **Most Important Feature**: Distance between stops
- **Dataset Split**: 80% training, 20% testing
- **Spark Configuration**: 4GB driver/executor memory for optimal performance

## 🔧 Script Features

### Flexible XML Parsing
- **Auto-discovery**: Automatically identifies XML structure
- **Namespace handling**: Properly handles TransXChange namespaces
- **Robust extraction**: Gracefully handles missing fields
- **Multiple schema support**: Works with various XML formats

### Data Quality Reporting
- Field-by-field completeness statistics
- ML readiness scoring
- Sample record display
- Missing data identification

### Error Handling
- Continues processing if individual files fail
- Reports errors without stopping entire batch
- Validates extracted data

## 📝 Sample Records

```
Line 107: Edgware → Edgware Station / Station Road
  Distance: 354m | Runtime: 60s | Speed: 21.2 km/h

Line 292: Ashley Drive / Farriers Way → Stirling Corner / Barnet Lane
  Distance: 405m | Runtime: 60s | Speed: 24.3 km/h

Line 331: Jackets Lane → Reservoir Road
  Distance: 2064m | Runtime: 75s | Speed: 99.1 km/h
```

## 🛠️ Technologies & Libraries

### Big Data Framework
- **Apache Spark 3.x**: Distributed computing framework
- **PySpark**: Python API for Spark
- **PySpark SQL**: DataFrame operations and transformations
- **PySpark MLlib**: Machine learning library (regression models)

### Python Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization (static plots)
- **seaborn**: Statistical data visualization
- **xml.etree.ElementTree**: XML parsing for TransXChange format

### Development Environment
- **Jupyter Notebook**: Interactive development and analysis
- **Python 3.x**: Core programming language
- **Windows PowerShell**: Terminal operations

## ⚠️ Limitations & Enrichment Opportunities

### Current Limitations
1. **No GPS Coordinates**: Stop locations not in XML
   - *Solution*: Enrich with external NaPTAN or OpenStreetMap data
   
2. **No Temporal Data**: No time-of-day or day-of-week
   - *Solution*: Add if available from other sources
   
3. **No Traffic Data**: No real-time conditions
   - *Solution*: Integrate with traffic APIs if needed

### Data Enrichment Ideas
1. Add stop coordinates from NaPTAN database
2. Include weather data (temperature, precipitation)
3. Add traffic congestion metrics
4. Include time-of-day features (peak vs off-peak)
5. Add day-of-week patterns

## 📚 Technical Details

### XML Structure (TransXChange Format)
```xml
<TransXChange>
  <StopPoints>
    <AnnotatedStopPointRef>
      <StopPointRef>490000070F</StopPointRef>
      <CommonName>Edgware</CommonName>
      ...
    </AnnotatedStopPointRef>
  </StopPoints>
  
  <RouteSections>
    <RouteLink>
      <From><StopPointRef>...</StopPointRef></From>
      <To><StopPointRef>...</StopPointRef></To>
      <Distance>354</Distance>
    </RouteLink>
  </RouteSections>
  
  <JourneyPatternSections>
    <JourneyPatternTimingLink>
      <From SequenceNumber="1">
        <StopPointRef>...</StopPointRef>
      </From>
      <To SequenceNumber="2">
        <StopPointRef>...</StopPointRef>
      </To>
      <RunTime>PT1M</RunTime>
    </JourneyPatternTimingLink>
  </JourneyPatternSections>
</TransXChange>
```

### Duration Parsing
ISO 8601 duration format (e.g., `PT1M30S`) is converted to seconds:
- `PT1M` → 60 seconds
- `PT50S` → 50 seconds
- `PT2M30S` → 150 seconds

## 🎯 Key Insights

### Strengths
✓ **Comprehensive**: All available fields extracted  
✓ **ML-Ready**: High data completeness (99.4%)  
✓ **Large Sample**: 40,000+ labeled examples  
✓ **Multiple Lines**: Diverse training data  
✓ **Clean Data**: Consistent formatting  

### Use Cases
1. ✅ Travel time prediction models
2. ✅ Route speed analysis
3. ✅ Service planning optimization
4. ✅ Performance benchmarking
5. ⚠️ Geographic analysis (needs coordinate enrichment)

## 🔄 Project Results & Achievements

### Data Processing Completed ✅
- **40,342 records** extracted from 7 XML files
- **532 unique stops** across the network
- **7 bus lines** (107, 292, 317, 324, 331, 491, 81)
- **99.4% ML readiness score** - excellent data quality
- **97.5% completeness** for runtime data

### Machine Learning Implementation ✅
- **3 regression models** trained and evaluated: Linear Regression, Random Forest, GBT
- **Distributed computing** leveraged Spark for scalable processing
- **Feature engineering** with categorical encoding and standardization
- **Pipeline automation** with end-to-end ML pipeline
- **Model comparison** using RMSE, MAE, R² metrics

### Key Deliverables
- ✓ `bus_data_comprehensive.csv`: Clean, ML-ready dataset
- ✓ `Analysis.ipynb`: Complete PySpark ML pipeline with visualizations
- ✓ `xml_to_csv_comprehensive.py`: Robust XML extraction script
- ✓ Model evaluation reports with comprehensive metrics

## 🚦 Future Enhancements

### Model Improvements
1. **Hyperparameter Tuning**: Use CrossValidator for optimal parameters
2. **Deep Learning**: Implement neural networks for complex patterns
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Real-time Prediction**: Deploy model as REST API service

### Data Enhancement
1. **GPS Coordinates**: Enrich with NaPTAN or OpenStreetMap data
2. **Weather Integration**: Add weather conditions (temperature, precipitation)
3. **Traffic Data**: Include real-time traffic congestion metrics
4. **Temporal Features**: Add time-of-day and day-of-week patterns
5. **Historical Trends**: Analyze seasonal patterns over time

### Deployment & Production
1. **Model Persistence**: Save trained models for reuse
2. **API Development**: Create prediction endpoints
3. **Dashboard**: Build interactive visualization dashboard
4. **Monitoring**: Implement model performance tracking and drift detection

## 🎓 Academic Context

- **Course**: ST5011CEM - Big Data Analytics
- **Institution**: 4th Semester Project
- **Focus**: Predictive analytics using Apache Spark and PySpark MLlib
- **Dataset Source**: Transport for London (TfL) / Metroline Travel Limited

## 📞 Support & Troubleshooting

### Common Issues
- **Spark Memory Errors**: Increase driver/executor memory in SparkSession configuration
- **Path Issues**: Update file paths to match your local directory structure
- **XML Parsing**: Ensure XML files are in TransXChange format
- **Missing Dependencies**: Install required packages with `pip install pyspark pandas numpy matplotlib seaborn`

### Getting Help
1. Check that XML files are in the `Metroline_TfL_9w6A35W/` directory
2. Verify Python 3.8+ and Java 8/11 are installed
3. Review console output for specific error messages
4. Ensure sufficient disk space for Spark operations

## 📚 References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [TransXChange Schema](https://www.gov.uk/government/collections/transxchange)
- [Transport for London Open Data](https://tfl.gov.uk/info-for/open-data-users/)

## 👤 Author

**Big Data Analytics Project**
- Course: ST5011CEM
- Date: February 2026

## 📄 License

This project is provided for educational and research purposes.
Bus data is property of Transport for London (TfL) and Metroline Travel Limited.

---
**Last Updated**: February 2026
