from rest_framework import serializers
from django.contrib.auth.hashers import make_password
from .models import UploadedFile, User
from NDISxRAG.settings import SUPER_KEY, LOGGER_NAME

import logging
logger = logging.getLogger(LOGGER_NAME)

class SignupSerializer(serializers.ModelSerializer):
    """
    Signup serializer: Serializes the data for user signup.
    """
    super_key = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'super_key', 'is_staff']
        extra_kwargs = {
            'password': {'write_only': True},
            'is_staff': {'default': False}
        }

    def validate_super_key(self, value):
        """
        Validates the super key.
        """
        if not SUPER_KEY:
            raise RuntimeError("SUPER_KEY is not set in the environment variables.")
        if value != SUPER_KEY:
            raise serializers.ValidationError("Invalid super key")
        return value

    def create(self, validated_data):
        """
        Creates a new user and returns a User instance.
        """
        logger.info("User data before creation: %s", validated_data)
        
        # Remove super_key (not a field in User)
        validated_data.pop('super_key', None)

        # Extract is_superuser flag (default: False)
        is_staff = validated_data.pop('is_staff', False)

        # Hash the password
        raw_password = validated_data.pop('password')
        hashed_password = make_password(raw_password)

        # Build the new user data
        validated_data['password'] = hashed_password
        validated_data["is_staff"] = is_staff

        # Create user
        user = User.objects.create(**validated_data)
        logger.info("Newly created user ID: %s", user.id)
        return user

class LoginSerializer(serializers.Serializer):
    """
    Login serializer: Serializes the data for user login
    """
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

class UpdateUserSerializer(serializers.ModelSerializer):
    """
    Update User serializer: Handles both reading user data and updating staff status
    """
    class Meta:
        model = User
        fields = ['id', 'is_staff', 'username']
        extra_kwargs = {
            'is_staff': {'required': True},
            'username': {'read_only': True},  # Read-only for updates, but included in GET responses
            'id': {'read_only': True}
        }

    def update(self, instance, validated_data):
        """
        Updates the user instance with the validated data.
        """
        logger.info("User data before update: %s", validated_data)
        instance.is_staff = validated_data.get('is_staff', instance.is_staff)
        instance.save()
        return instance

class UploadedFileSerializer(serializers.ModelSerializer):
    """
    UploadedFile serializer: Serializes the data for uploaded filess
    """
    class Meta:
        model = UploadedFile
        fields = ['id', 'user', 'file', 'filename', 'file_type', 'uploaded_at', 'last_processed_step', 'finished_processing']
        read_only_fields = ['id', 'uploaded_at']

    def create(self, validated_data):
        return super().create(validated_data)