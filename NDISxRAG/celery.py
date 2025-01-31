# NDISxRAG/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NDISxRAG.settings')
app = Celery('NDISxRAG')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
