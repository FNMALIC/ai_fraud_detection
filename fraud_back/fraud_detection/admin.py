from django.contrib import admin
from .models import Transaction, UserProfile, ModelInfo, SystemHealth

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['transaction_id', 'user', 'type', 'amount', 'fraud_prediction', 'risk_level', 'timestamp']
    list_filter = ['type', 'fraud_prediction', 'risk_level', 'timestamp']
    search_fields = ['transaction_id', 'user__username', 'name_orig', 'name_dest']
    readonly_fields = ['transaction_id', 'timestamp', 'processed_at']

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'phone_number', 'location', 'is_verified', 'created_at']
    list_filter = ['is_verified', 'location', 'created_at']
    search_fields = ['user__username', 'user__email', 'phone_number']

@admin.register(ModelInfo)
class ModelInfoAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'version', 'auc_score', 'accuracy', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']

@admin.register(SystemHealth)
class SystemHealthAdmin(admin.ModelAdmin):
    list_display = ['status', 'message', 'timestamp']
    list_filter = ['status', 'timestamp']
    readonly_fields = ['timestamp']