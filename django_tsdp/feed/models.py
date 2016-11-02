from django.db import models
#from custom_user.models import EmailUser
from django.contrib.auth.models import User
from taggit.managers import TaggableManager
from taggit.models import TaggedItemBase
from datetime import *
from django.template.defaultfilters import slugify
from django.contrib.humanize.templatetags.humanize import *

from autoslug import AutoSlugField
from django.utils.safestring import mark_safe
from dateutil.relativedelta import relativedelta
import dateutil.parser
import pytz
import os
import re
import uuid
import operator

mytz = pytz.timezone('Asia/Seoul')

class Instrument(models.Model):
    broker=models.CharField(max_length=255, null=True, db_index=True)
    sym=models.CharField(max_length=255, null=True, db_index=True)
    cur=models.CharField(max_length=255, null=True, db_index=True)
    exch=models.CharField(max_length=255, null=True, db_index=True)
    secType=models.CharField(max_length=255, null=True, db_index=True)
    trade_freq=models.IntegerField(null=True, db_index=True)
    mult=models.IntegerField(null=True, db_index=True)
    local_sym=models.CharField(max_length=255, null=True, db_index=True)
    
    contractMonth=models.IntegerField(null=True, db_index=True)
    expiry=models.CharField(max_length=255, null=True, db_index=True)
    evRule=models.CharField(max_length=255, null=True, db_index=True)
    liquidHours=models.CharField(max_length=255, null=True, db_index=True)
    longName=models.CharField(max_length=255, null=True, db_index=True)
    minTick=models.FloatField(max_length=255, null=True, db_index=True)
    timeZoneId=models.IntegerField(null=True, db_index=True)
    tradingHours=models.CharField(max_length=255, null=True, db_index=True)
    underConId=models.IntegerField(null=True, db_index=True)
    
    created_at = models.DateTimeField(
        auto_now_add=True, null=True, db_index=True)
    updated_at = models.DateTimeField(
        auto_now_add=True, null=True, db_index=True)
    crawl_source=models.CharField(max_length=200, default='', blank=True, null=True)

    def __unicode__(self):
        return '%s' % self.name

    def save(self, *args, **kwargs):
        if self.created_at == None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
        super(Instrument, self).save(*args, **kwargs)


class System(models.Model):
    #user = models.OneToOneField(User, primary_key=True)
    version= models.CharField(max_length=255, null=True, db_index=True)
    system= models.CharField(max_length=255, null=True, db_index=True)
    name=models.CharField(max_length=255, null=True, db_index=True)
    c2id=models.CharField(max_length=255, null=True, db_index=True)
    c2api=models.CharField(max_length=255, null=True, db_index=True)
    c2qty=models.IntegerField(null=True, db_index=True)
    c2submit=models.BooleanField(default=False)
    c2instrument=models.ForeignKey(Instrument, related_name='c2instrument', null=True, db_index=True)
    ibqty=models.IntegerField(null=True, db_index=True)
    ibinstrument=models.ForeignKey(Instrument, related_name='ibinstrument', null=True, db_index=True)
    ibsubmit=models.BooleanField(default=False)
    trade_freq=models.IntegerField(null=True, db_index=True)
    ibmult=models.IntegerField(null=True, db_index=True)
    c2mult=models.IntegerField(null=True, db_index=True)
    signal=models.CharField(max_length=255, null=True, db_index=True)

class Feed(models.Model):
    instrument=models.ForeignKey(Instrument, db_index=True)
    frequency=models.IntegerField(null=True, db_index=True)
    date=models.DateTimeField(
        null=True, db_index=True)
    open=models.FloatField(null=True)
    high=models.FloatField(null=True)
    low=models.FloatField(null=True)
    close=models.FloatField(null=True)
    volume=models.FloatField(null=True)
    
    created_at = models.DateTimeField(
        auto_now_add=True, null=True, db_index=True)
    updated_at = models.DateTimeField(
        auto_now_add=True, null=True, db_index=True)
    crawl_source=models.CharField(max_length=200, default='', blank=True, null=True)

    def __unicode__(self):
        return '%s' % self.name

    def save(self, *args, **kwargs):
        if self.created_at == None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
        super(Feed, self).save(*args, **kwargs)



