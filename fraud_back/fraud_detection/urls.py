from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'transactions', views.TransactionViewSet, basename='transaction')
router.register(r'profiles', views.UserProfileViewSet, basename='userprofile')

urlpatterns = [
    path('', include(router.urls)),
    path('health/status/', views.health_status, name='health_status'),
    path('health/model-info/', views.model_info, name='model_info'),
    path('fraud/predict/', views.predict_fraud, name='predict_fraud'),
    path('fraud/predict-batch/', views.predict_fraud_batch, name='predict_fraud_batch'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
]
