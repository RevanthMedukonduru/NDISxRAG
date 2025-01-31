from django.contrib.auth.hashers import check_password
from django.core.files.base import ContentFile
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated

from .models import User, UploadedFile
from .serializer import SignupSerializer, LoginSerializer, UploadedFileSerializer, UpdateUserSerializer
from .tasks import process_excel_task
from agents.agent import generate_response

# Logger
import logging
from NDISxRAG.settings import LOGGER_NAME
logger = logging.getLogger(LOGGER_NAME)

class SignupView(generics.CreateAPIView):
    """
    Handles user signup requests.
    """
    serializer_class = SignupSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        """
        Override the create method from CreateAPIView to provide a
        custom response after user creation.
        """
        try:
            logger.info("Signup request received")
            serializer = self.get_serializer(data=request.data)
            logger.info("L1")
            serializer.is_valid(raise_exception=True)
            logger.info("L2")
            user = serializer.save()
            logger.info("L3")
            # Generate a token for the new user
            logger.info("user type: %s", type(user))
            _, created = Token.objects.get_or_create(user=user)
            logger.info("User created successfully | Username: %s | Token Created: %s", request.data["username"], created)
            return Response({"message": "Signup successful"}, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error("Error creating user: %s", str(e))
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class UpdateStaffView(generics.GenericAPIView):
    """
    View for updating user staff status and retrieving all user profiles if the authenticated user is staff.
    """
    serializer_class = UpdateUserSerializer
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        POST: Update the staff status of a user using the UpdateUserSerializer.
        """
        if not request.user.is_staff:
            return Response(
                {"error": "You do not have permission to update staff status"}, 
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            user = User.objects.get(id=request.data.get("id"))
        except User.DoesNotExist:
            return Response(
                {"error": "Invalid user"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        serializer = self.serializer_class(user, data=request.data, partial=True)
        if serializer.is_valid():
            updated_user = serializer.save()
            return Response({
                "message": "User staff status updated",
                "id": updated_user.id,
                "is_staff": updated_user.is_staff,
                "username": updated_user.username
            }, status=status.HTTP_200_OK)
        
        return Response(
            {"error": serializer.errors}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    def get(self, request):
        """
        GET: Retrieve all user profiles if the authenticated user is a staff member.
        """
        if not request.user.is_staff:
            return Response({"error": "Only staff members can view user profiles"}, status=status.HTTP_403_FORBIDDEN)

        # Fetch and serialize all users
        users = User.objects.all()
        serializer = self.serializer_class(users, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)    

class LoginView(generics.GenericAPIView):
    """
    Handles user login requests.
    """
    serializer_class = LoginSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        """
        Validates the user's credentials and returns a success or error response.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        username = serializer.validated_data["username"]
        password = serializer.validated_data["password"]

        try:
            logger.debug("Login request received")
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            logger.error("User not found | Username: %s", username)
            return Response(
                {"error": "Invalid username or User not found"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not check_password(password, user.password):
            logger.error("Invalid password | Username: %s", username)
            return Response(
                {"error": "Invalid password"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        try:
            # Token generation and replacement
            Token.objects.filter(user=user).delete()  # Deleting existing token
            new_token = Token.objects.create(user=user)  # Created a new token

            logger.info("Login successful | Username: %s", username)
            return Response(
                {
                    "message": "Login successful",
                    "is_staff": user.is_staff,
                    "token": new_token.key,
                },
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error("Error logging in | Error: %s", str(e))
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class FileUploadView(generics.GenericAPIView):
    """
    Handles file upload requests from Streamlit.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """
        Handle file upload from Streamlit.
        Expect 'username' in form-data and 'file' in files.
        """
        logger.debug("File upload request received")
        username = request.POST.get("username")
        file_obj = request.FILES.get("file")  # 'file' must match Streamlit form key

        if not username or not file_obj:
            logger.error("Username or file missing")
            return Response({"error": "Username or file missing"}, status=status.HTTP_400_BAD_REQUEST)

        # Get user object
        try:
            logger.debug("Getting user object | Username: %s", username)
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            logger.error("Invalid user | Username: %s", username)
            return Response({"error": "Invalid user"}, status=status.HTTP_400_BAD_REQUEST)

        # Prepare data for serializer
        data = request.data.copy()
        data["user"] = user.id
        data["filename"] = file_obj.name
        data["file_type"] = file_obj.content_type

        # We can directly pass 'file': file_obj to the serializer if using DRF's File Upload
        serializer = UploadedFileSerializer(data=data)
        if serializer.is_valid():
            logger.info("File uploaded | Filename: %s | User: %s", file_obj.name, username)
            instance = serializer.save(file=file_obj)  # Save the file to disk

            # Trigger background task (Celery or Django async) (if its Excel file only)
            if file_obj.content_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                logger.info("PDF file uploaded, BG task triggered | Filename: %s | User: %s", file_obj.name, username)
                process_excel_task.delay(instance.id)  # pass the ID to the async task
                # or if using Django's built-in async:
                # process_pdf_task(instance.id)  # define as async def

            logger.info("Finished processing file | Filename: %s | User: %s", file_obj.name, username)
            return Response({"message": "File uploaded", "file_id": instance.id},
                            status=status.HTTP_201_CREATED)
        else:
            logger.error(serializer.errors)
            logger.error("Error uploading file | Filename: %s | User: %s", file_obj.name, username)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FileListView(generics.GenericAPIView):
    """
    Returns a list of uploaded files for a user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """
        Return a list of uploaded files for a user specified via query param: ?username=<user>
        """
        username = request.GET.get("username")
        if not username:
            return Response({"error": "Username missing"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({"error": "Invalid user"}, status=status.HTTP_400_BAD_REQUEST)

        # Get all files for this user
        files = UploadedFile.objects.filter(user=user).order_by('-uploaded_at')
        serializer = UploadedFileSerializer(files, many=True)

        # Return minimal info in JSON (filename, file_type, etc.)
        data = serializer.data
        return Response(data, status=status.HTTP_200_OK)

class ConversationView(generics.CreateAPIView):
    """
    Handle user conversation requests:
      - Accepts a query and chat history from the client.
      - Finds the most recently uploaded file for the authenticated user.
      - Generates a response by passing file path, query, and chat history to your utility function.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        POST endpoint to process a conversation query.

        Body Parameters:
            query (str): The user's query or message.
            chat_history (list): A list of chat messages so far.
        
        Returns:
            200 OK with JSON: {"response": <generated_response>}
            400 Bad Request if user has no uploaded file.
        """
        logger.info("Conversation request received")
        user = request.user
        original_query = request.data.get("query")
        chat_history = request.data.get("chat_history", [])

        # Grab the most recent file uploaded by the current user
        uploaded_file = UploadedFile.objects.filter(user=user).order_by('-uploaded_at').first()
        if not uploaded_file:
            return Response({"error": "No file uploaded yet"}, status=status.HTTP_400_BAD_REQUEST)

        # Convert FieldFile to a string path if needed
        file_path = str(uploaded_file.file)

        # Perform your domain logic or call a helper function
        response_text, citations = generate_response(
            user_id=user.id,
            file_path=file_path,
            original_query=original_query,
            chat_history=chat_history
        )
        return Response({"response": response_text, "citations": citations}, status=status.HTTP_200_OK) 
        