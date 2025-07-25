from django.urls import path
from . import views

urlpatterns = [
    path('', views.main_page, name='main'),
    path('registration/', views.registration, name='registration'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
