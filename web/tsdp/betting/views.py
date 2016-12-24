from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import UserSelection
from .helpers import *
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.views.static import serve
import pandas as pd
import sqlite3
import json
import os

dbPath = '/tsdpWEB/tsdp/db.sqlite3'
readConn = sqlite3.connect(dbPath)

def downloaddb(request):
    filepath = '/ml-tsdp/data/futures.sqlite3'
    return serve(request, os.path.basename(filepath), os.path.dirname(filepath))

# Create your views here.
#def refreshMetaData(request):
#    updateMeta = MetaData(mcdate=MCdate(), timestamp=getTimeStamp())
#    updateMeta.save()

def addrecord(request):
    record = UserSelection(userID=request.GET['user_id'], selection=request.GET['Selection'], \
                           v4futures=request.GET['v4futures'], v4mini=request.GET['v4mini'], \
                           v4micro=request.GET['v4micro'], mcdate=MCdate(), timestamp=getTimeStamp())
    record.save()
    return HttpResponse(json.dumps({"id": record.id}))


def getrecords(request):
    # records = [ dict((cn, getattr(data, cn)) for cn in ('v4futures', 'v4mini')) for data in UserSelection.objects.all() ]
    # print(records)
    # return HttpResponse(json.dumps(records))

    firstrec = UserSelection.objects.order_by('-timestamp').first()
    firstdata = firstrec.dic()
    # print(json.dumps(firstdata))
    recent = UserSelection.objects.order_by('-timestamp')[:20]
    recentdata = [dict((cn, getattr(data, cn)) for cn in ('timestamp', 'mcdate', 'selection')) for data in recent]
    return HttpResponse(json.dumps({"first": firstdata, "recent": recentdata}))


def post_list(request):
    updateMeta()
    getAccountValues()
    return render(request, 'index.html', {})

def markets(request):
    return render(request, 'symbols.html', {})

def futures(request):
    return render(request, 'futures.html', {})

def system_charts(request, symbol):
    imagedir = '/ml-tsdp/web/tsdp/betting/static/images/'
    filenames = [x for x in os.listdir(imagedir) if 'v4_'+symbol+'_' in x]
    return render(request, 'system_charts.html', {'charts':filenames})

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


def last_userselection(request):
    #lastSelection = pd.read_sql('select * from betting_userselection where timestamp=\
    #        (select max(timestamp) from betting_userselection as maxtimestamp)', con=readConn, index_col='userID')
    lastSelection=UserSelection.objects.all().order_by('-timestamp')[0]
    return JsonResponse(lastSelection.dic())
