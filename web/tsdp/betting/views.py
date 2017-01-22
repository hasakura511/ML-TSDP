from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import UserSelection, get_blends
from .helpers import *
from .start_immediate import start_immediate
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import JsonResponse
from django.views.static import serve
import pandas as pd
import sqlite3
import json
import os

#dbPath = '/tsdpWEB/tsdp/db.sqlite3'
#readConn = sqlite3.connect(dbPath)

def downloaddb(request):
    filepath = '/ml-tsdp/data/futures.sqlite3'
    return serve(request, os.path.basename(filepath), os.path.dirname(filepath))

# Create your views here.
#def refreshMetaData(request):
#    updateMeta = MetaData(mcdate=MCdate(), timestamp=getTimeStamp())
#    updateMeta.save()

def addrecord(request):
    userselection=UserSelection()
    json_cloc=request.GET['componentloc']
    #json_cloc = userselection.json_cloc
    _, list_boxstyles = get_blends(cloc=userselection.cloc)
    json_boxstyles=json.dumps(list_boxstyles)

    #for user customized styles (maybe later)
    #list_boxstyles = json.loads(request.GET['boxstyles'])
    #_, list_boxstyles = UserSelection.get_blends(json.loads(json_cloc), list_boxstyles)
    #json_boxstyles=json.dumps(list_boxstyles)
    with open('/ml-tsdp/web/tsdp/performance_data.json', 'r') as f:
        json_performance = json.load(f)

    record = UserSelection(userID=request.GET['user_id'], selection=request.GET['Selection'], \
                            v4futures=request.GET['v4futures'], v4mini=request.GET['v4mini'], \
                            v4micro=request.GET['v4micro'],
                            componentloc = json_cloc,
                            boxstyles = json_boxstyles,
                            performance = json_performance,
                            mcdate=MCdate(), timestamp=getTimeStamp())
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

def index(request):
    return render(request, 'loading_page.html', {})

#def loading_page(request):
#    return render(request, 'loading_page.html', {})

def board(request):
    lastSelection = UserSelection.objects.all().order_by('-timestamp').first()

    #check if any immediate orders
    if 'True' in [order[1] for sys, order in eval(lastSelection.selection).items()]:
        print('processing immediate orders')
        start_immediate()

    updateMeta()
    getAccountValues()
    return render(request, 'board.html', {})

def getmetadata(request):
    returnrec = MetaData.objects.order_by('-timestamp').first()
    returndata = returnrec.dic()
    print(returndata)
    return HttpResponse(json.dumps(returndata))

def getaccountdata(request):
    returnrec = AccountData.objects.order_by('-timestamp').first()
    returndata = returnrec.dic()
    print(returndata)
    return HttpResponse(json.dumps(returndata))

def symbols(request):
    futuresdict = get_futures_dictionary()
    return render(request, 'symbols.html', {'groups':futuresdict})

def futures(request, date=None):
    context={}
    if date==None:
        context['accounts']=get_overview()
        date=context['accounts']['v4futures']
        context['date']=dt.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        context['archive'] = False
    else:
        context['accounts']={}
        accounts = ['v4micro', 'v4mini', 'v4futures']
        for account in accounts:
            context['accounts'][account] = date.replace('-','')
        context['date']=date
        context['archive']=True
    return render(request, 'futures2.html', context)

def timetable(request):
    eastern = timezone('US/Eastern')
    now = dt.now(get_localzone()).astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S %p EST')

    return render(request, 'timetable.html', {'time':now, 'timetable':get_detailed_timetable()})

def order_status(request):
    context={'logfiles':[]}
    context['slippagedates'], context['orderstatus']=get_order_status()
    class LogFiles(object):
        def __init__(self, filename):
            self.filename = filename

        def display_text_file(self):
            with open(self.filename) as fp:
                return fp.read()

    files = get_logfiles(search_string='moc_live')
    timestamp = sorted([f[-21:] for f in files])[-1]
    logfile=[f for f in files if 'error' not in f and timestamp in f][-1]
    context['logfiles'].append((logfile,LogFiles(logfile)))

    errorfiles= [f for f in files if logfile[-21:] in f and 'error' in f]
    if len(errorfiles)>0:
        errorfile=errorfiles[-1]
        context['logfiles'].append((errorfile,LogFiles(errorfile)))

    return render(request, 'show_log.html', context)

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

def archive(request):
    return render(request, 'archive.html', {'dates':archive_dates()})