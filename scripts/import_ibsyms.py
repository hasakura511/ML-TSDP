# -*- coding: utf-8 -*- 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from django.db import models

sys.path.append("../")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tsdp.settings")
import tsdp
import tsdp.settings as bsettings
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


def read_contract_csv(filename):

    
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        rownum=0
        for row in reader:
            rownum+=1
            if rownum > 1:
                (contractMonth,currency,evMultiplier,evRule,exchange,expiry,liquidHours,longName,minTick,secType,symbol,timeZoneId,tradingHours,underConId)=row
                inst_list=Instrument.objects.filter(sym=symbol).filter(contractMonth=contractMonth).filter(secType=secType)
                if inst_list and len(inst_list) > 0:
                    inst=inst_list[0]
                else:
                    inst=Instrument()
                inst.broker='ib'
                inst.contractMonth=contractMonth
                inst.cur=currency
                inst.mult=float(evMultiplier)
                inst.evRule=evRule
                inst.exch=exchange
                inst.expiry=expiry
                inst.liquidHours=liquidHours
                inst.longName=longName
                inst.minTick=minTick
                inst.secType=secType
                inst.sym=symbol
                inst.local_sym=symbol
                inst.timeZoneId=timeZoneId
                inst.tradingHours=tradingHours
                inst.underConId=underConId
                inst.save()
                print "Saving ",inst.sym
                    
                    
                print row;

read_contract_csv('../data/systems/ib_contracts.csv')
