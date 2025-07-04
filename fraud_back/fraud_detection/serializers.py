from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Transaction, UserProfile, ModelInfo, SystemHealth

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = ['user', 'phone_number', 'location', 'is_verified', 'created_at']

class TransactionSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Transaction
        fields = '__all__'
        read_only_fields = ['transaction_id', 'user', 'fraud_probability', 
                           'fraud_prediction', 'risk_level', 'timestamp', 'processed_at']

class TransactionInputSerializer(serializers.Serializer):
    step = serializers.IntegerField(min_value=1)
    type = serializers.ChoiceField(choices=Transaction.TRANSACTION_TYPES)
    amount = serializers.DecimalField(max_digits=15, decimal_places=2, min_value=0)
    nameOrig = serializers.CharField(max_length=50)
    oldbalanceOrig = serializers.DecimalField(max_digits=15, decimal_places=2, min_value=0)
    newbalanceOrig = serializers.DecimalField(max_digits=15, decimal_places=2, min_value=0)
    nameDest = serializers.CharField(max_length=50)
    oldbalanceDest = serializers.DecimalField(max_digits=15, decimal_places=2, min_value=0)
    newbalanceDest = serializers.DecimalField(max_digits=15, decimal_places=2, min_value=0)

class BatchTransactionSerializer(serializers.Serializer):
    transactions = TransactionInputSerializer(many=True)

class ModelInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelInfo
        fields = '__all__'


class SystemHealthSerializer(serializers.ModelSerializer):
    class Meta:
        model = SystemHealth
        fields = '__all__'