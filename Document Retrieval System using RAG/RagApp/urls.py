from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),   # ✅ ADD THIS LINE

    path("index.html", views.index, name="index"),
    path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
    path("Register.html", views.Register, name="Register"),
    path("Contactus.html", views.Contactus, name="Contactus"),
    path("Aboutus.html", views.Aboutus, name="Aboutus"),        
    path("RegisterAction", views.RegisterAction, name="RegisterAction"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    
    path("UploadDocument.html", views.UploadDocument, name="UploadDocument"),        
    path("UploadDocumentAction", views.UploadDocumentAction, name="UploadDocumentAction"),
    path("Retrieval.html", views.Retrieval, name="Retrieval"),  
    path("RetrievalAction", views.RetrievalAction, name="RetrievalAction"),  
    path("Generation.html", views.Generation, name="Generation"),  
    path("GenerationAction", views.GenerationAction, name="GenerationAction"),  
    path("DownloadFile", views.DownloadFile, name="DownloadFile"),  
]
