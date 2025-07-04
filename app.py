# Enhanced Model Compatibility API with Full Swagger Documentation
import os
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.exceptions import BadRequest
from flask_cors import CORS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompatibleModelManager:
    """
    Enhanced model manager that handles version compatibility issues
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
        df = self.create_mobile_money_features(df)
        
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
    
    def create_mobile_money_features(self, df):
        """
        Create engineered features for mobile money transactions
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
        Predict fraud probability for a transaction
        """
        if not self.model_package['is_trained']:
            logger.warning("Model not trained! Training with sample data...")
            self.train_model_with_sample_data()
        
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([transaction_data])
            
            # Apply feature engineering
            df_input = self.create_mobile_money_features(df_input)
            
            # Encode transaction type
            df_input['type_encoded'] = self.model_package['label_encoder'].transform(df_input['type'])
            
            # Select features
            X_input = df_input[self.feature_columns]
            
            # Handle infinite values
            X_input = X_input.replace([np.inf, -np.inf], np.nan)
            X_input = X_input.fillna(0)
            
            # Scale features if needed
            if self.model_package['model_name'] == 'Logistic Regression':
                X_input_scaled = self.model_package['scaler'].transform(X_input)
                fraud_prob = self.model_package['model'].predict_proba(X_input_scaled)[0, 1]
            else:
                fraud_prob = self.model_package['model'].predict_proba(X_input)[0, 1]
            
            return float(fraud_prob)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def save_model(self):
        """
        Save the current model package
        """
        try:
            joblib.dump(self.model_package, self.model_path)
            logger.info(f"✅ Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

# Initialize Flask app
app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False  # Disable field masking for better Swagger docs
CORS(app)  # Enable CORS for all routes

# Initialize Flask-RESTX with enhanced configuration
api = Api(
    app,
    version='1.0',
    title='Mobile Money Fraud Detection API - Enhanced',
    description='AI-powered fraud detection for Cameroon mobile money transactions (Orange Money & MTN MoMo) with compatibility fixes',
    doc='/docs/',  # Swagger UI will be at /docs/
    prefix='/api/v1'
)

# Create namespaces
ns_fraud = api.namespace('fraud', description='Fraud Detection Operations')
ns_health = api.namespace('health', description='Health Check Operations')
ns_model = api.namespace('model', description='Model Management Operations')

# Initialize model manager
try:
    model_manager = CompatibleModelManager()
    logger.info("✅ Model Manager initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize Model Manager: {e}")
    model_manager = None

# Define API models for Swagger documentation
transaction_model = api.model('Transaction', {
    'step': fields.Integer(required=True, description='Time step of transaction', example=1),
    'type': fields.String(required=True, description='Transaction type', 
                          enum=['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'],
                          example='TRANSFER'),
    'amount': fields.Float(required=True, description='Transaction amount in XAF', example=5000.0),
    'nameOrig': fields.String(required=True, description='Origin account ID', example='C1231006815'),
    'oldbalanceOrig': fields.Float(required=True, description='Origin account balance before transaction', example=10000.0),
    'newbalanceOrig': fields.Float(required=True, description='Origin account balance after transaction', example=5000.0),
    'nameDest': fields.String(required=True, description='Destination account ID', example='C1666544295'),
    'oldbalanceDest': fields.Float(required=True, description='Destination account balance before transaction', example=0.0),
    'newbalanceDest': fields.Float(required=True, description='Destination account balance after transaction', example=5000.0)
})

fraud_prediction_model = api.model('FraudPrediction', {
    'transaction_id': fields.String(description='Unique transaction identifier'),
    'fraud_probability': fields.Float(description='Probability of fraud (0-1)'),
    'fraud_prediction': fields.String(description='Fraud prediction result'),
    'risk_level': fields.String(description='Risk level classification'),
    'timestamp': fields.String(description='Prediction timestamp'),
    'model_version': fields.String(description='Model version used')
})

batch_transaction_model = api.model('BatchTransactions', {
    'transactions': fields.List(fields.Nested(transaction_model), required=True, 
                               description='List of transactions to analyze')
})

health_status_model = api.model('HealthStatus', {
    'status': fields.String(description='API health status'),
    'timestamp': fields.String(description='Check timestamp'),
    'model_status': fields.String(description='Model loading status'),
    'model_name': fields.String(description='Loaded model name'),
    'model_auc_score': fields.Float(description='Model AUC score')
})

model_info_model = api.model('ModelInfo', {
    'model_name': fields.String(description='Model algorithm name'),
    'auc_score': fields.Float(description='Model AUC score'),
    'feature_count': fields.Integer(description='Number of features'),
    'features': fields.List(fields.String, description='List of feature names'),
    'supported_transaction_types': fields.List(fields.String, description='Supported transaction types'),
    'is_trained': fields.Boolean(description='Whether model is trained')
})

# Health check endpoints
@ns_health.route('/status')
class HealthCheck(Resource):
    @ns_health.marshal_with(health_status_model)
    def get(self):
        """Check API health status"""
        try:
            # Check if model is loaded
            model_status = "loaded" if model_manager and model_manager.model_package is not None else "not_loaded"
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_status": model_status,
                "model_name": model_manager.model_package['model_name'] if model_manager and model_manager.model_package else None,
                "model_auc_score": float(model_manager.model_package['auc_score']) if model_manager and model_manager.model_package else None
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}, 500

@ns_model.route('/info')
class ModelInfo(Resource):
    @ns_model.marshal_with(model_info_model)
    def get(self):
        """Get detailed model information"""
        try:
            if not model_manager or model_manager.model_package is None:
                return {"error": "Model not loaded"}, 500
            
            return {
                "model_name": model_manager.model_package['model_name'],
                "auc_score": float(model_manager.model_package['auc_score']),
                "feature_count": len(model_manager.model_package['feature_columns']),
                "features": model_manager.model_package['feature_columns'],
                "supported_transaction_types": ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'],
                "is_trained": model_manager.model_package.get('is_trained', False)
            }
        except Exception as e:
            return {"error": str(e)}, 500

@ns_model.route('/train')
class TrainModel(Resource):
    def post(self):
        """Train model with sample data (for testing and development)"""
        if not model_manager:
            return {'error': 'Model manager not available'}, 500
        
        try:
            model_manager.train_model_with_sample_data()
            return {
                'message': 'Model training completed successfully',
                'timestamp': datetime.now().isoformat(),
                'model_name': model_manager.model_package['model_name'],
                'auc_score': float(model_manager.model_package['auc_score'])
            }, 200
        except Exception as e:
            logger.error(f"Training endpoint error: {e}")
            return {'error': str(e)}, 500

# Fraud detection endpoints
@ns_fraud.route('/predict')
class FraudPrediction(Resource):
    @ns_fraud.expect(transaction_model)
    @ns_fraud.marshal_with(fraud_prediction_model)
    def post(self):
        """Predict fraud probability for a single transaction"""
        if not model_manager:
            return {'error': 'Model manager not available'}, 500
            
        try:
            transaction_data = request.json
            
            # Validate required fields
            required_fields = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrig', 
                             'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
            
            for field in required_fields:
                if field not in transaction_data:
                    raise BadRequest(f"Missing required field: {field}")
            
            # Validate transaction type
            valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
            if transaction_data['type'] not in valid_types:
                raise BadRequest(f"Invalid transaction type. Must be one of: {valid_types}")
            
            # Make prediction
            fraud_prob = model_manager.predict_fraud(transaction_data)
            fraud_prediction = "FRAUD" if fraud_prob > 0.5 else "NORMAL"
            
            # Determine risk level
            if fraud_prob >= 0.8:
                risk_level = "HIGH"
            elif fraud_prob >= 0.5:
                risk_level = "MEDIUM"
            elif fraud_prob >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            result = {
                "transaction_id": transaction_data.get('nameOrig', '') + "_" + str(transaction_data.get('step', '')),
                "fraud_probability": round(fraud_prob, 4),
                "fraud_prediction": fraud_prediction,
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat(),
                "model_version": model_manager.model_package['model_name']
            }
            
            # Log high-risk transactions
            if fraud_prob >= 0.5:
                logger.warning(f"High-risk transaction detected: {result}")
            
            return result
            
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": "Internal server error"}, 500

@ns_fraud.route('/predict-batch')
class BatchFraudPrediction(Resource):
    @ns_fraud.expect(batch_transaction_model)
    def post(self):
        """Predict fraud probability for multiple transactions"""
        if not model_manager:
            return {'error': 'Model manager not available'}, 500
            
        try:
            data = request.json
            transactions = data.get('transactions', [])
            
            if not transactions:
                raise BadRequest("No transactions provided")
            
            if len(transactions) > 1000:  # Limit batch size
                raise BadRequest("Maximum 1000 transactions per batch")
            
            results = []
            
            for i, transaction_data in enumerate(transactions):
                try:
                    # Validate required fields
                    required_fields = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrig', 
                                     'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
                    
                    for field in required_fields:
                        if field not in transaction_data:
                            results.append({
                                "transaction_index": i,
                                "error": f"Missing required field: {field}"
                            })
                            continue
                    
                    # Make prediction
                    fraud_prob = model_manager.predict_fraud(transaction_data)
                    fraud_prediction = "FRAUD" if fraud_prob > 0.5 else "NORMAL"
                    
                    # Determine risk level
                    if fraud_prob >= 0.8:
                        risk_level = "HIGH"
                    elif fraud_prob >= 0.5:
                        risk_level = "MEDIUM"
                    elif fraud_prob >= 0.2:
                        risk_level = "LOW"
                    else:
                        risk_level = "MINIMAL"
                    
                    results.append({
                        "transaction_index": i,
                        "transaction_id": transaction_data.get('nameOrig', '') + "_" + str(transaction_data.get('step', '')),
                        "fraud_probability": round(fraud_prob, 4),
                        "fraud_prediction": fraud_prediction,
                        "risk_level": risk_level
                    })
                    
                except Exception as e:
                    results.append({
                        "transaction_index": i,
                        "error": str(e)
                    })
            
            # Summary statistics
            successful_predictions = [r for r in results if 'fraud_probability' in r]
            fraud_count = len([r for r in successful_predictions if r['fraud_prediction'] == 'FRAUD'])
            high_risk_count = len([r for r in successful_predictions if r['risk_level'] == 'HIGH'])
            
            response = {
                "timestamp": datetime.now().isoformat(),
                "total_transactions": len(transactions),
                "successful_predictions": len(successful_predictions),
                "fraud_detected": fraud_count,
                "high_risk_transactions": high_risk_count,
                "results": results
            }
            
            return response
            
        except BadRequest as e:
            return {"error": str(e)}, 400
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return {"error": "Internal server error"}, 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Test the system
    print("🧪 Testing the enhanced system...")
    
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
        
        try:
            fraud_prob = model_manager.predict_fraud(test_transaction)
            print(f"✅ Test prediction result: {fraud_prob}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("🚀 Starting Enhanced Flask application...")
    print("📚 Swagger documentation available at: http://localhost:5000/docs/")
    app.run(debug=True, host='0.0.0.0', port=5000)