# Customer Churn Prediction with Artificial Neural Networks

A comprehensive machine learning project that predicts customer churn using Artificial Neural Networks (ANN). This project includes data preprocessing, model training, evaluation, and a web-based prediction interface using Streamlit.

## 🚀 Features

- **Neural Network Model**: Deep learning model built with TensorFlow/Keras
- **Data Preprocessing**: Automated feature engineering and scaling
- **Interactive Web App**: Streamlit-based user interface for real-time predictions
- **Model Persistence**: Saved models and preprocessors for easy deployment
- **Comprehensive Analysis**: Jupyter notebooks for data exploration and model evaluation
- **TensorBoard Integration**: Training visualization and monitoring

## 📊 Project Structure

```
ANN/
├── app.py                          # Streamlit web application
├── model.h5                        # Trained neural network model
├── scaler.pkl                      # StandardScaler for feature normalization
├── label_encode_gender.pkl         # LabelEncoder for gender encoding
├── onehot_encoder_geo.pkl          # OneHotEncoder for geography encoding
├── Churn_Modelling.csv             # Original dataset
├── experiments.ipynb               # Model training and experimentation
├── predictions.ipynb               # Prediction analysis and evaluation
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
└── logs/                          # TensorBoard training logs
```

## 📋 Prerequisites

- Python 3.10+
- Jupyter Notebook
- Required Python packages (see requirements.txt):
  - `tensorflow` - Deep learning framework
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning utilities
  - `tensorboard` - Training visualization
  - `matplotlib` - Data visualization
  - `streamlit` - Web application framework

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ANN
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Running the Web Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - Enter customer information using the interactive form
   - Get instant churn predictions

### Model Training and Experimentation

1. **Open the experiments notebook**:
   ```bash
   jupyter notebook experiments.ipynb
   ```

2. **Run the prediction analysis**:
   ```bash
   jupyter notebook predictions.ipynb
   ```

### Key Components

#### 1. Data Preprocessing
```python
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Categorical encoding
label_encoder = LabelEncoder()
gender_encoded = label_encoder.fit_transform(gender)

# One-hot encoding
onehot_encoder = OneHotEncoder()
geo_encoded = onehot_encoder.fit_transform(geography)
```

#### 2. Neural Network Architecture
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
```

#### 3. Web Application Features
- **Interactive Input Forms**: User-friendly interface for data entry
- **Real-time Predictions**: Instant churn probability calculations
- **Visual Feedback**: Clear probability scores and predictions

## 🔍 Model Details

### Input Features
- **Credit Score**: Customer's credit rating
- **Geography**: Customer's location (France, Spain, Germany)
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years with the bank
- **Balance**: Account balance
- **Number of Products**: Products subscribed to
- **Has Credit Card**: Credit card ownership (0/1)
- **Is Active Member**: Active membership status (0/1)
- **Estimated Salary**: Customer's estimated salary

### Model Architecture
- **Input Layer**: 11 features (after preprocessing)
- **Hidden Layer 1**: 6 neurons with ReLU activation
- **Hidden Layer 2**: 6 neurons with ReLU activation
- **Output Layer**: 1 neuron with Sigmoid activation
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 32
- **Validation Split**: 0.2
- **Early Stopping**: Patience = 10 epochs

## 📈 Performance Metrics

The model achieves the following performance:
- **Accuracy**: High classification accuracy
- **Precision**: Precision for churn prediction
- **Recall**: Recall for churn detection
- **F1-Score**: Balanced precision and recall

## 🔧 Customization

### Adding New Features
```python
# Add new features to the input data
input_data['NewFeature'] = [new_value]
```

### Modifying Model Architecture
```python
# Custom neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=12, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
```

### Changing Hyperparameters
```python
# Custom training parameters
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

## 📊 Data Analysis

### Dataset Overview
- **Total Records**: 10,000+ customer records
- **Features**: 10 original features
- **Target**: Binary churn prediction (0/1)
- **Data Quality**: Clean, preprocessed dataset

### Feature Engineering
- **Scaling**: StandardScaler for numerical features
- **Encoding**: LabelEncoder for gender, OneHotEncoder for geography
- **Feature Selection**: All relevant features included

## 🚀 Deployment

### Local Deployment
```bash
# Run the Streamlit app locally
streamlit run app.py
```

### Cloud Deployment
The application can be deployed on:
- **Heroku**: Using the provided runtime.txt
- **Streamlit Cloud**: Direct deployment from GitHub
- **AWS/GCP**: Containerized deployment

### Model Serving
```python
# Load saved model and preprocessors
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
```

## 📈 Monitoring and Logging

### TensorBoard Integration
```bash
# View training logs
tensorboard --logdir=logs
```

### Model Performance Tracking
- Training and validation loss curves
- Accuracy metrics over time
- Model convergence analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with TensorFlow/Keras for deep learning
- Web interface powered by Streamlit
- Data preprocessing with scikit-learn
- Visualization with TensorBoard and matplotlib

## 📞 Support

If you have any questions or need help with the implementation, please open an issue on GitHub.

---

**Happy Predicting! 🎯**
