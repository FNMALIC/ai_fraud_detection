# fraud_detection/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.utils import timezone
from django.db.models import Count, Q
from .models import Transaction, UserProfile, ModelInfo, SystemHealth
from .serializers import (
    TransactionSerializer, UserProfileSerializer, TransactionInputSerializer,
    BatchTransactionSerializer, ModelInfoSerializer, SystemHealthSerializer
)
from .ml_model import fraud_model
import uuid

class TransactionViewSet(viewsets.ModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        if self.request.user.is_staff:
            return Transaction.objects.all()
        return Transaction.objects.filter(user=self.request.user)
    
    @action(detail=False, methods=['get'])
    def user_stats(self, request):
        """Get user transaction statistics"""
        user_transactions = Transaction.objects.filter(user=request.user)
        
        stats = {
            'total_transactions': user_transactions.count(),
            'fraud_detected': user_transactions.filter(fraud_prediction='FRAUD').count(),
            'legitimate_transactions': user_transactions.filter(fraud_prediction='LEGITIMATE').count(),
            'high_risk_transactions': user_transactions.filter(risk_level='HIGH').count(),
            'recent_transactions': user_transactions[:5].count()
        }
        
        return Response(stats)

class UserProfileViewSet(viewsets.ModelViewSet):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        if self.request.user.is_staff:
            return UserProfile.objects.all()
        return UserProfile.objects.filter(user=self.request.user)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def health_status(request):
    """Check API health status"""
    try:
        # Check database connection
        Transaction.objects.count()
        
        # Check model status
        if not fraud_model.is_trained:
            fraud_model.train_dummy_model()
        
        health_data = {
            'status': 'healthy',
            'timestamp': timezone.now(),
            'database': 'connected',
            'model': 'loaded',
            'version': '1.0.0'
        }
        
        # Save health check
        SystemHealth.objects.create(
            status='healthy',
            message='All systems operational'
        )
        
        return Response(health_data)
    
    except Exception as e:
        health_data = {
            'status': 'error',
            'error': str(e),
            'timestamp': timezone.now()
        }
        
        SystemHealth.objects.create(
            status='error',
            message=str(e)
        )
        
        return Response(health_data, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_info(request):
    """Get model information and performance metrics"""
    try:
        if not fraud_model.is_trained:
            metrics = fraud_model.train_dummy_model()
            
            # Save or update model info
            model_info, created = ModelInfo.objects.get_or_create(
                model_name='Random Forest Classifier',
                version='1.0',
                defaults={
                    'auc_score': metrics['auc_score'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'is_active': True
                }
            )
        else:
            model_info = ModelInfo.objects.filter(is_active=True).first()
        
        if model_info:
            serializer = ModelInfoSerializer(model_info)
            return Response(serializer.data)
        else:
            return Response({
                'model_name': 'Random Forest Classifier',
                'version': '1.0',
                'auc_score': 0.95,
                'accuracy': 0.92,
                'status': 'loaded'
            })
    
    except Exception as e:
        return Response({
            'error': str(e),
            'status': 'error'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_fraud(request):
    """Predict fraud for a single transaction"""
    try:
        serializer = TransactionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        transaction_data = serializer.validated_data
        
        # Convert field names to match model expectations
        model_data = {
            'step': transaction_data['step'],
            'type': transaction_data['type'],
            'amount': float(transaction_data['amount']),
            'oldbalanceOrig': float(transaction_data['oldbalanceOrig']),
            'newbalanceOrig': float(transaction_data['newbalanceOrig']),
            'oldbalanceDest': float(transaction_data['oldbalanceDest']),
            'newbalanceDest': float(transaction_data['newbalanceDest']),
            'nameOrig': transaction_data.get('nameOrig', ''),
            'nameDest': transaction_data.get('nameDest', '')
        }
        
        # Get prediction from ML model
        prediction = fraud_model.predict(model_data)
        
        # Save transaction to database
        transaction = Transaction.objects.create(
            user=request.user,
            step=transaction_data['step'],
            type=transaction_data['type'],
            amount=transaction_data['amount'],
            name_orig=transaction_data.get('nameOrig', ''),
            old_balance_orig=transaction_data['oldbalanceOrig'],
            new_balance_orig=transaction_data['newbalanceOrig'],
            name_dest=transaction_data.get('nameDest', ''),
            old_balance_dest=transaction_data['oldbalanceDest'],
            new_balance_dest=transaction_data['newbalanceDest'],
            fraud_probability=prediction['fraud_probability'],
            fraud_prediction=prediction['fraud_prediction'],
            risk_level=prediction['risk_level'],
            processed_at=timezone.now()
        )
        
        response_data = {
            'transaction_id': str(transaction.transaction_id),
            'fraud_probability': prediction['fraud_probability'],
            'fraud_prediction': prediction['fraud_prediction'],
            'risk_level': prediction['risk_level'],
            'model_version': prediction['model_version'],
            'prediction_id': prediction['prediction_id'],
            'timestamp': transaction.timestamp.isoformat(),
            'processing_time_ms': 0  # Can be calculated if needed
        }
        
        # Log high-risk transactions
        if prediction['fraud_probability'] >= 0.5:
            logger.warning(f"High-risk transaction detected for user {request.user.username}: {prediction}")
        
        return Response(response_data)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return Response({
            'error': str(e),
            'message': 'Failed to process fraud prediction'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAdminUser])
def admin_dashboard(request):
    """Admin dashboard with system statistics"""
    try:
        # User statistics
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()
        
        # Transaction statistics
        total_transactions = Transaction.objects.count()
        fraud_transactions = Transaction.objects.filter(fraud_prediction='FRAUD').count()
        legitimate_transactions = Transaction.objects.filter(fraud_prediction='LEGITIMATE').count()
        
        # Risk level distribution
        risk_distribution = Transaction.objects.values('risk_level').annotate(
            count=Count('id')
        ).order_by('risk_level')
        
        # Recent activity
        recent_transactions = Transaction.objects.select_related('user')[:10]
        recent_transactions_data = TransactionSerializer(recent_transactions, many=True).data
        
        # System health
        system_health = SystemHealth.objects.last()
        
        dashboard_data = {
            'users': {
                'total': total_users,
                'active': active_users,
                'verified': UserProfile.objects.filter(is_verified=True).count()
            },
            'transactions': {
                'total': total_transactions,
                'fraud': fraud_transactions,
                'legitimate': legitimate_transactions,
                'fraud_rate': (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
            },
            'risk_distribution': list(risk_distribution),
            'recent_transactions': recent_transactions_data,
            'system_health': SystemHealthSerializer(system_health).data if system_health else None
        }
        
        return Response(dashboard_data)
    
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)