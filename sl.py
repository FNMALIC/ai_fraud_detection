# Model Compatibility Fix for Local Deployment
import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompatibleModelManager:
    """
    Model manager that handles version compatibility issues
    """
    
    def __init__(self, model_path="mobile_money_fraud_model.pkl"):
        self.model_path = model_path
        self.model_package = None
        self.feature_columns = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """
        Try to load existing model, create new one if incompatible
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Attempting to load model from {self.model_path}")
                self.model_package = joblib.load(self.model_path)
                logger.info("✅ Model loaded successfully!")
                self.feature_columns = self.model_package.get('feature_columns', [])
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Creating a new compatible model...")
                self.create_compatible_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating a new compatible model due to compatibility issues...")
            self.create_compatible_model()
    
    def create_compatible_model(self):
        """
        Create a new model with current scikit-learn version
        """
        logger.info("🔧 Creating new compatible model...")
        
        # Create a basic model package with your current scikit-learn version
        label_encoder = LabelEncoder()
        label_encoder.fit(['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'])
        
        # Feature columns based on your training
        self.feature_columns = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
            'newbalanceDest', 'hour', 'sender_freq', 'receiver_freq', 'balance_ratio_orig', 
            'balance_ratio_dest', 'amount_to_balance_ratio', 'is_round_amount', 
            'zero_balance_orig', 'zero_balance_dest', 'balance_diff_orig', 
            'balance_diff_dest', 'type_encoded'
        ]
        
        # Create a new Gradient Boosting model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model_package = {
            'model': model,
            'scaler': StandardScaler(),
            'label_encoder': label_encoder,
            'feature_columns': self.feature_columns,
            'model_name': 'Gradient Boosting',
            'auc_score': 0.95,  # Placeholder
            'feature_importance': None,
            'is_trained': False  # Flag to indicate if model needs training
        }
        
        logger.info("✅ Compatible model package created!")
        logger.warning("⚠️ Model needs to be trained with actual data for production use")
    
    def create_sample_training_data(self, n_samples=10000):
        """
        Create sample training data for demonstration
        """
        logger.info("🎲 Creating sample training data...")
        
        np.random.seed(42)
        
        # Generate synthetic mobile money transaction data
        data = {
            'step': np.random.randint(1, 744, n_samples),  # 1 month of hours
            'type': np.random.choice(['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'], 
                                   n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05]),
            'amount': np.random.lognormal(8, 1.5, n_samples),  # Log-normal distribution for amounts
            'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
            'oldbalanceOrig': np.random.lognormal(9, 1.2, n_samples),
            'newbalanceOrig': np.random.lognormal(8.5, 1.3, n_samples),
            'nameDest': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
            'oldbalanceDest': np.random.lognormal(8, 1.5, n_samples),
            'newbalanceDest': np.random.lognormal(8.2, 1.4, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (simplified logic)
        fraud_conditions = (
            ((df['type'] == 'TRANSFER') & (df['amount'] > 200000)) |  # Large transfers
            ((df['type'] == 'CASH_OUT') & (df['amount'] > 100000)) |  # Large cash outs
            (df['amount'] > 500000)  # Very large amounts
        )
        
        df['isFraud'] = fraud_conditions.astype(int)
        
        # Add some random fraud cases
        random_fraud = np.random.choice(df.index, size=int(0.001 * len(df)), replace=False)
        df.loc[random_fraud, 'isFraud'] = 1
        
        logger.info(f"✅ Sample data created: {len(df)} transactions, {df['isFraud'].sum()} fraudulent")
        return df
    
    def train_model_with_sample_data(self):
        """
        Train the model with sample data
        """
        if self.model_package['is_trained']:
            logger.info("Model is already trained!")
            return
        
        logger.info("🎯 Training model with sample data...")
        
        # Create sample data
        df = self.create_sample_training_data()
        
        # Feature engineering
        df = self.create_complete_features(df)
        
        # Encode categorical variables
        df['type_encoded'] = self.model_package['label_encoder'].transform(df['type'])
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['isFraud']
        
        # Handle missing values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        self.model_package['model'].fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model_package['model'].predict(X_test)
        y_prob = self.model_package['model'].predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        self.model_package['auc_score'] = auc_score
        self.model_package['is_trained'] = True
        
        logger.info(f"✅ Model training completed!")
        logger.info(f"📊 AUC Score: {auc_score:.4f}")
        logger.info(f"📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the trained model
        self.save_model()
    
    def create_complete_features(self, df):
        """
        Create all required features for the model
        """
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['step'] % 24
        df['day'] = df['step'] // 24
        df['is_weekend'] = (df['day'] % 7).isin([5, 6]).astype(int)
        
        # Fix column name issue
        df['oldbalanceOrg'] = df['oldbalanceOrig']
        
        # Balance difference features
        df['balance_diff_orig'] = df['oldbalanceOrig'] - df['newbalanceOrig']
        df['balance_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Balance ratio features
        df['balance_ratio_orig'] = np.where(df['oldbalanceOrig'] > 0, 
                                           df['newbalanceOrig'] / df['oldbalanceOrig'], 0)
        df['balance_ratio_dest'] = np.where(df['oldbalanceDest'] > 0, 
                                           df['newbalanceDest'] / df['oldbalanceDest'], 0)
        
        # Amount to balance ratio
        df['amount_to_balance_ratio'] = np.where(df['oldbalanceOrig'] > 0, 
                                                df['amount'] / df['oldbalanceOrig'], 0)
        
        # Frequency features (simplified for sample data)
        sender_counts = df['nameOrig'].value_counts()
        receiver_counts = df['nameDest'].value_counts()
        df['sender_freq'] = df['nameOrig'].map(sender_counts).fillna(1)
        df['receiver_freq'] = df['nameDest'].map(receiver_counts).fillna(1)
        
        # Round amount feature
        df['is_round_amount'] = (df['amount'] % 1000 == 0).astype(int)
        
        # Zero balance flags
        df['zero_balance_orig'] = (df['newbalanceOrig'] == 0).astype(int)
        df['zero_balance_dest'] = (df['oldbalanceDest'] == 0).astype(int)
        
        return df
    
    def predict_fraud(self, transaction_data):
        """
        Predict fraud for a given transaction
        """
        if not self.model_package['is_trained']:
            logger.warning("Model not trained! Training with sample data...")
            self.train_model_with_sample_data()
        
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([transaction_data])
            
            # Apply feature engineering
            df_input = self.create_complete_features(df_input)
            
            # Encode transaction type
            df_input['type_encoded'] = self.model_package['label_encoder'].transform(df_input['type'])
            
            # Select features
            X_input = df_input[self.feature_columns]
            
            # Handle missing values
            X_input = X_input.fillna(0)
            X_input = X_input.replace([np.inf, -np.inf], 0)
            
            # Predict
            fraud_prob = self.model_package['model'].predict_proba(X_input)[0, 1]
            fraud_prediction = fraud_prob > 0.5
            
            return {
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(fraud_prediction),
                'confidence': 'high' if abs(fraud_prob - 0.5) > 0.3 else 'medium',
                'model_info': {
                    'model_name': self.model_package['model_name'],
                    'auc_score': self.model_package['auc_score'],
                    'is_trained': self.model_package['is_trained']
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'fraud_probability': 0.5,
                'is_fraud': False,
                'confidence': 'low',
                'error': str(e)
            }
    
    def save_model(self):
        """
        Save the current model package
        """
        try:
            joblib.dump(self.model_package, self.model_path)
            logger.info(f"✅ Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

# Flask Application
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, title='Mobile Money Fraud Detection API', version='1.0', 
          description='Fixed API for fraud detection with compatibility handling')

# Initialize model manager
try:
    model_manager = CompatibleModelManager()
    logger.info("✅ Model Manager initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize Model Manager: {e}")
    model_manager = None

# API Models
transaction_model = api.model('Transaction', {
    'step': fields.Integer(required=True, description='Time step'),
    'type': fields.String(required=True, description='Transaction type', 
                          enum=['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']),
    'amount': fields.Float(required=True, description='Transaction amount'),
    'nameOrig': fields.String(required=True, description='Origin account'),
    'oldbalanceOrig': fields.Float(required=True, description='Original balance before transaction'),
    'newbalanceOrig': fields.Float(required=True, description='New balance after transaction'),
    'nameDest': fields.String(required=True, description='Destination account'),
    'oldbalanceDest': fields.Float(required=True, description='Destination balance before'),
    'newbalanceDest': fields.Float(required=True, description='Destination balance after')
})

@api.route('/predict')
class FraudPrediction(Resource):
    @api.expect(transaction_model)
    def post(self):
        """Predict if a transaction is fraudulent"""
        if not model_manager:
            return {'error': 'Model not available'}, 500
        
        try:
            data = request.get_json()
            result = model_manager.predict_fraud(data)
            return result, 200
        except Exception as e:
            logger.error(f"Prediction endpoint error: {e}")
            return {'error': str(e)}, 500

@api.route('/health')
class HealthCheck(Resource):
    def get(self):
        """Health check endpoint"""
        if model_manager and model_manager.model_package:
            return {
                'status': 'healthy',
                'model_loaded': True,
                'model_trained': model_manager.model_package.get('is_trained', False),
                'model_name': model_manager.model_package.get('model_name', 'Unknown')
            }, 200
        return {'status': 'unhealthy', 'model_loaded': False}, 500

@api.route('/train')
class TrainModel(Resource):
    def post(self):
        """Train model with sample data (for testing)"""
        if not model_manager:
            return {'error': 'Model manager not available'}, 500
        
        try:
            model_manager.train_model_with_sample_data()
            return {'message': 'Model training completed successfully'}, 200
        except Exception as e:
            logger.error(f"Training endpoint error: {e}")
            return {'error': str(e)}, 500

if __name__ == '__main__':
    # Test the system
    print("🧪 Testing the fixed system...")
    
    if model_manager:
        # Test transaction
        test_transaction = {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 181.00,
            'nameOrig': 'C1231006815',
            'oldbalanceOrig': 181.00,
            'newbalanceOrig': 0.00,
            'nameDest': 'C1666544295',
            'oldbalanceDest': 0.00,
            'newbalanceDest': 0.00
        }
        
        result = model_manager.predict_fraud(test_transaction)
        print(f"✅ Test prediction result: {result}")
    
    print("🚀 Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
