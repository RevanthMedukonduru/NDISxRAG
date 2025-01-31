from django.urls import path
from app import views

urlpatterns = [
    path('signup/', views.SignupView.as_view(), name='signup'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('upload-file/', views.FileUploadView.as_view(), name='upload-file'),
    path('list-files/', views.FileListView.as_view(), name='list-files'),
    path('chat/', views.ConversationView.as_view(), name='chat'),
    path('update-staff/', views.UpdateStaffView.as_view(), name='update-staff'),
]