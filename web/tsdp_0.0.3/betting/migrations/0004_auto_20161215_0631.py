# -*- coding: utf-8 -*-
# Generated by Django 1.10.4 on 2016-12-15 12:31
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('betting', '0003_accountdata_metadata'),
    ]

    operations = [
        migrations.RenameField(
            model_name='accountdata',
            old_name='accountvalues',
            new_name='value1',
        ),
        migrations.RenameField(
            model_name='accountdata',
            old_name='urpnls',
            new_name='value2',
        ),
    ]
