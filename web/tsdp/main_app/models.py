from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Treasure(models.Model):
    name = models.CharField(max_length=100)
    value = models.DecimalField(max_digits=10, decimal_places=2)
    material = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    img_url = models.CharField(max_length=100)
    
    def __str__(self):
        return self.name
        
class Dictionary(models.Model):
    class Meta:
        db_table = 'Dictionary' # This tells Django where the SQL table is
        managed = False
    CSIsym = models.CharField(max_length=100)
    Desc = models.CharField(max_length=100)
    def __str__(self):
        return self.CSIsym
        
class currenciesATR(models.Model):
    class Meta:
        db_table = 'currenciesATR' # This tells Django where the SQL table is
        managed = False
    pairs = models.CharField(max_length=100)
    Last = models.CharField(max_length=100)
    def __str__(self):
        return self.pairs