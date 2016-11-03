"""
Default application configuration.
In use as of Django 1.7.
"""
from django.apps import AppConfig
#from watson import search as watson

class FeedConfig(AppConfig):
    name = 'feed'

    #def ready(self):
       