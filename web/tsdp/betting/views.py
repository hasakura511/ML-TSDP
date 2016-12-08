from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import UserSelection
from django import forms
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm

import time
import math
import datetime
from datetime import datetime as dt
from pytz import timezone
from tzlocal import get_localzone

import json


# Create your views here.
class LoginForm(forms.Form):
    username = forms.CharField(label='User Name', max_length=64)
    password = forms.CharField(widget=forms.PasswordInput())


def MCdate():
    cutoff = datetime.time(17, 0, 0, 0)
    cutoff2 = datetime.time(23, 59, 59)
    eastern = timezone('US/Eastern')
    now = dt.now(get_localzone())
    now = now.astimezone(eastern)

    if now.weekday() == 5:
        # Saturday so set to monday
        next = now + datetime.timedelta(days=2)
        return next.strftime("%Y%m%d")

    if now.weekday() == 6:
        # Sunday so set to monday
        next = now + datetime.timedelta(days=1)
        return next.strftime("%Y%m%d")

    if now.time() > cutoff and now.time() < cutoff2:
        if now.weekday() > 4:
            if now.weekday() == 4:
                # friday after cutoff so set to monday
                next = now + datetime.timedelta(days=3)
                return next.strftime("%Y%m%d")
        else:
            # M-TH after cutoff
            next = now + datetime.timedelta(days=1)
            return next.strftime("%Y%m%d")
    else:
        # M-FRI before cutoff?
        return now.strftime("%Y%m%d")


def getTimeStamp():
    timestamp = int(time.mktime(dt.utcnow().timetuple()))
    return timestamp


def addrecord(request):
    record = UserSelection(userID=request.GET['user_id'], selection=request.GET['Selection'],
                           v4futures=request.GET['v4futures'], v4mini=request.GET['v4mini'],
                           v4micro=request.GET['v4micro'], mcdate=MCdate(), timestamp=getTimeStamp())
    record.save()
    return HttpResponse(json.dumps({"id": record.id}))


def getrecords(request):
    records = [dict((cn, getattr(data, cn)) for cn in ('name', 'data')) for data in MyTable.objects.all()]
    print(records)
    return HttpResponse(json.dumps(records))


def post_list(request):
    return render(request, 'index.html', {})


def profile(request, username):
    user = User.objects.get(username=username)
    return render(request, 'profile.html', {'username': username})


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            u = form.cleaned_data['username']
            p = form.cleaned_data['password']
            user = authenticate(username=u, password=p)
            if user is not None:
                if user.is_active:
                    login(request, user)
                    return HttpResponseRedirect('/')
                else:
                    print('The account has been disabled.')
                    return HttpResponseRedirect('/')
            else:
                print('The username and password were incorrect.')
                return HttpResponseRedirect('/')


    else:
        form = LoginForm()
        return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    return HttpResponseRedirect('/')


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/login/')
    else:
        form = UserCreationForm()
        return render(request, 'registration.html', {'form': form})
