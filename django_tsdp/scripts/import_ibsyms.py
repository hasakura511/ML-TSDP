# -*- coding: utf-8 -*- 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from django.db import models

sys.path.append("../")
sys.path.append("../../")

sys.path.append("../")
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tsdp.settings")
import tsdp
import tsdp.settings as settings
from feed.models import *
import datetime

import csv

def output_csv(filename, queryset):

    opts = queryset.model._meta
    model = queryset.model
    f = open(filename, 'w')
    writer = csv.writer(f)
    field_names = [field.name for field in opts.fields]
    # Write a first row with header information
    writer.writerow(field_names)
    # Write data rows
    for obj in queryset:
        writer.writerow([getattr(obj, field) for field in field_names])
    f.close()


def read_csv(filename):

    
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            (contractMonth,currency,evMultiplier,evRule,exchange,expiry,liquidHours,longName,minTick,secType,symbol,timeZoneId,tradingHours,underConId)=row
            inst_list=Instrument.objects.filter(symbol=symbol).filter(contractMonth=contractMonth).filter(secType=secType)
            if inst_list and len(inst_list) > 0:
                inst=inst_list[0]
            else:
                inst=Instrument()
                inst.contractMonth=contractMonth
                
            print row;

read_csv('../../data/systems/ib_contracts.csv')
