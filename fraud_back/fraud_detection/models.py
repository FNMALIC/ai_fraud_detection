from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, blank=True)
    location = models.CharField(max_length=100, default='Douala, Cameroon')
    is_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - Profile"

class Transaction(models.Model):
    TRANSACTION_TYPES = [
        ('TRANSFER', 'Transfer'),
        ('CASH_OUT', 'Cash Out'),
        ('PAYMENT', 'Payment'),
        ('CASH_IN', 'Cash In'),
        ('DEBIT', 'Debit'),
    ]
    
    RISK_LEVELS = [
        ('MINIMAL', 'Minimal'),
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
    ]
    
    FRAUD_PREDICTIONS = [
        ('LEGITIMATE', 'Legitimate'),
        ('FRAUD', 'Fraud'),
    ]
    
    # Transaction identification
    transaction_id = models.UUIDField(default=uuid.uuid4, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='transactions')
    
    # Transaction details
    step = models.IntegerField(help_text="Time step in simulation")
    type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    amount = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Origin account
    name_orig = models.CharField(max_length=50, help_text="Origin account name")
    old_balance_orig = models.DecimalField(max_digits=15, decimal_places=2)
    new_balance_orig = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Destination account
    name_dest = models.CharField(max_length=50, help_text="Destination account name")
    old_balance_dest = models.DecimalField(max_digits=15, decimal_places=2)
    new_balance_dest = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Fraud detection results
    fraud_probability = models.FloatField(null=True, blank=True)
    fraud_prediction = models.CharField(max_length=20, choices=FRAUD_PREDICTIONS, null=True, blank=True)
    risk_level = models.CharField(max_length=20, choices=RISK_LEVELS, null=True, blank=True)
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.username} - {self.type} - {self.amount} XAF"

class ModelInfo(models.Model):
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    auc_score = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.model_name} v{self.version}"

class SystemHealth(models.Model):
    status = models.CharField(max_length=20, default='healthy')
    message = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"System Health - {self.status} at {self.timestamp}"