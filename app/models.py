from django.db import models
from django.contrib.auth.models import AbstractUser

# User model
class User(AbstractUser):
    """
    User model: Stores user details
    """
    def __str__(self):
        return self.username

class UploadedFile(models.Model):
    """
    UploadedFile model: Stores uploaded files
    
    Fields:
    user: ForeignKey
    file: FileField
    filename: CharField
    file_type: CharField
    uploaded_at: DateTimeField
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')  # Adjust upload_to path to your preference
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    last_processed_step = models.CharField(max_length=100, null=True, blank=True)
    finished_processing = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.filename} ({self.file_type}) - {self.user.username}"