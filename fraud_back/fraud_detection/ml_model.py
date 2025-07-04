# fraud_detection/ml_model.py
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
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from django.conf import settings
import uuid

# Setup logging
logger = logging.getLogger(__name__)

class DjangoCompatibleModelManager:
    """
    Django-compatible model manager for fraud detection
    """
    
    def __init__(self):
        # Use Django's media directory for model storage
        self.model_path = os.path.join(settings.MEDIA_ROOT, 'models', 'mobile_money_fraud_model.pkl')
        self.model_package = None
        self.feature_columns = None
        self.is_trained = False
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """
        Try to load existing model, create new one if incompatible
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Attempting to load model from {self.model_path}")
                self.model_package = joblib.load(self.model_path)
                self.is_trained = self.model_package.get('is_trained', False)
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
        
        # Create a new Random Forest model (changed from GradientBoosting for better compatibility)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model_package = {
            'model': model,
            'scaler': StandardScaler(),
            'label_encoder': label_encoder,
            'feature_columns': self.feature_columns,
            'model_name': 'Random Forest Classifier',
            'auc_score': 0.95,  # Placeholder
            'accuracy': 0.92,   # Placeholder
            'precision': 0.89,  # Placeholder
            'recall': 0.94,     # Placeholder
            'f1_score': 0.91,   # Placeholder
            'feature_importance': None,
            'is_trained': False,
            'version': '1.0',
            'created_at': datetime.now().isoformat()
        }
        
        self.is_trained = False
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
    
    def train_dummy_model(self):
        """
        Train the model with sample data (for Django compatibility)
        """
        if self.is_trained:
            logger.info("Model is already trained!")
            return self.get_model_metrics()
        
        logger.info("🎯 Training model with sample data...")
        
        try:
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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model_package['model'].fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model_package['model'].predict(X_test)
            y_prob = self.model_package['model'].predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_prob)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Update model package with metrics
            self.model_package.update({
                'auc_score': auc_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'is_trained': True,
                'trained_at': datetime.now().isoformat()
            })
            
            self.is_trained = True
            
            logger.info(f"✅ Model training completed!")
            logger.info(f"📊 AUC Score: {auc_score:.4f}")
            logger.info(f"📊 Accuracy: {accuracy:.4f}")
            logger.info(f"📊 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"📊 F1-Score: {f1:.4f}")
            
            # Save the trained model
            self.save_model()
            
            return self.get_model_metrics()
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def get_model_metrics(self):
        """
        Get current model performance metrics
        """
        if not self.model_package:
            return {}
        
        return {
            'auc_score': self.model_package.get('auc_score', 0),
            'accuracy': self.model_package.get('accuracy', 0),
            'precision': self.model_package.get('precision', 0),
            'recall': self.model_package.get('recall', 0),
            'f1_score': self.model_package.get('f1_score', 0),
            'model_name': self.model_package.get('model_name', 'Unknown'),
            'version': self.model_package.get('version', '1.0'),
            'is_trained': self.is_trained
        }
    
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
    
    def predict(self, transaction_data):
        """
        Predict fraud probability for a transaction (Django-compatible method)
        """
        if not self.is_trained:
            logger.warning("Model not trained! Training with sample data...")
            self.train_dummy_model()
        
        try:
            # Create DataFrame from input
            df_input = pd.DataFrame([transaction_data])
            
            # Apply feature engineering
            df_input = self.create_mobile_money_features(df_input)
            
            # Encode transaction type
            if 'type' in transaction_data:
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
            
            # Determine prediction and risk level
            fraud_prediction = "FRAUD" if fraud_prob > 0.5 else "LEGITIMATE"
            
            if fraud_prob >= 0.8:
                risk_level = "HIGH"
            elif fraud_prob >= 0.5:
                risk_level = "MEDIUM"
            elif fraud_prob >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"
            
            result = {
                'fraud_probability': float(fraud_prob),
                'fraud_prediction': fraud_prediction,
                'risk_level': risk_level,
                'model_version': self.model_package['model_name'],
                'prediction_id': str(uuid.uuid4())
            }
            
            # Log high-risk transactions
            if fraud_prob >= 0.5:
                logger.warning(f"High-risk transaction detected: Probability={fraud_prob:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, transactions_list):
        """
        Predict fraud for multiple transactions
        """
        results = []
        
        for i, transaction_data in enumerate(transactions_list):
            try:
                prediction = self.predict(transaction_data)
                prediction['transaction_index'] = i
                results.append(prediction)
            except Exception as e:
                results.append({
                    'transaction_index': i,
                    'error': str(e),
                    'fraud_probability': None,
                    'fraud_prediction': 'ERROR',
                    'risk_level': 'UNKNOWN'
                })
        
        return results
    
    def save_model(self):
        """
        Save the current model package
        """
        try:
            joblib.dump(self.model_package, self.model_path)
            logger.info(f"✅ Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if not self.is_trained or not hasattr(self.model_package['model'], 'feature_importances_'):
            return {}
        
        importance_dict = dict(zip(
            self.feature_columns,
            self.model_package['model'].feature_importances_
        ))
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def model_summary(self):
        """
        Get complete model summary
        """
        return {
            'model_info': {
                'name': self.model_package.get('model_name', 'Unknown') if self.model_package else 'Not Loaded',
                'version': self.model_package.get('version', '1.0') if self.model_package else 'Unknown',
                'is_trained': self.is_trained,
                'created_at': self.model_package.get('created_at') if self.model_package else None,
                'trained_at': self.model_package.get('trained_at') if self.model_package else None
            },
            'performance_metrics': self.get_model_metrics(),
            'feature_info': {
                'feature_count': len(self.feature_columns) if self.feature_columns else 0,
                'features': self.feature_columns if self.feature_columns else [],
                'feature_importance': self.get_feature_importance()
            },
            'supported_types': ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
        }

# Create a global instance for Django compatibility
fraud_model = DjangoCompatibleModelManager()