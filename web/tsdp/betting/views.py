from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import UserSelection
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
from os.path import isfile, join
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
    json_cloc=request.GET['componentloc']
    cloc=json.loads(json_cloc)
    print 'cloc\n',cloc
    #get_blends(cloc=json.loads(json_cloc))
    #cloc= [{'c0':'Off'},{'c1':'RiskOn'},{'c2':'None'},{'c3':'None'},{'c4':'None'},{'c5':'None'},{'c6':'None'},{'c7':'None'},{'c8':'None'},{'c9':'None'},{'c10':'None'},{'c11':'None'},{'c12':'None'},{'c13':'None'},{'c14':'None'},]
    #list_boxstyles=[{'c0':{'text':'Off','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'225823','fill-R':'34','fill-G':'88','fill-B':'35','filename':''}},{'c1':{'text':'RiskOn','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'BE0032','fill-R':'190','fill-G':'0','fill-B':'50','filename':''}},{'c2':{'text':'RiskOff','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'222222','fill-R':'34','fill-G':'34','fill-B':'34','filename':''}},{'c3':{'text':'LowestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F38400','fill-R':'243','fill-G':'132','fill-B':'0','filename':''}},{'c4':{'text':'HighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'FFFF00','fill-R':'255','fill-G':'255','fill-B':'0','filename':''}},{'c5':{'text':'AntiHighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'A1CAF1','fill-R':'161','fill-G':'202','fill-B':'241','filename':''}},{'c6':{'text':'Anti50/50','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'C2B280','fill-R':'194','fill-G':'178','fill-B':'128','filename':''}},{'c7':{'text':'Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'E68FAC','fill-R':'230','fill-G':'143','fill-B':'172','filename':''}},{'c8':{'text':'Anti-Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F99379','fill-R':'249','fill-G':'147','fill-B':'121','filename':''}},{'c9':{'text':'Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'654522','fill-R':'101','fill-G':'69','fill-B':'34','filename':''}},{'c10':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c11':{'text':'Anti-Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'008856','fill-R':'0','fill-G':'136','fill-B':'86','filename':''}},{'c12':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c13':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c14':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_clear_all':{'text':'Clear All Bets','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_create_new':{'text':'Create New Board','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_confirm_orders':{'text':'Save Orders','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_ok':{'text':'Enter Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_order_cancel':{'text':'Cancel','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_order_active':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_inactive':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_save_ok':{'text':'Place Immediate Orders Now','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_save_cancel':{'text':'OK/Change Immediate Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_order_dialog':{'text':'<b>MOC:</b> Market-On-Close Order. New signals are generated at the close of the market will be placed as Market Orders before the close.<br><b>Immediate:</b> Immediate uses signals generated as of the last Market Close.  If the market is closed, order will be placed as Market-On-Open orders. Otherwise, it will be placed as Market Orders. At the next trigger time, new signals will be placed as MOC orders.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_save_dialog':{'text':'<center><b>Orders successfully saved.</b><br></center> MOC orders will be placed at the trigger times. If you have entered any immediate orders you may place them now or you may cancel and save different orders.  After the page is refreshed you can check order status to see if the orders were placed.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'text_table':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_table_title':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_datetimenow':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_triggertimes':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance_account':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'chip_v4micro':{'text':'50K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_maroon.png'}},{'chip_v4mini':{'text':'100K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_purple.png'}},{'chip_v4futures':{'text':'250K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_orange.png'}},]
    #list_boxstyles=[{'c0':{'text':'Off','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'BE0032','fill-R':'34','fill-G':'88','fill-B':'35','filename':''}},{'c1':{'text':'RiskOn','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'c2':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c3':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c4':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c5':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c6':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c7':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c8':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c9':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c10':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c11':{'text':'','text-color':'FFFFFF','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c12':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c13':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'c14':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_clear_all':{'text':'Clear Board','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_create_new':{'text':'New Board','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_confirm_orders':{'text':'Process Orders','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'24','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_ok':{'text':'Enter Orders','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_order_cancel':{'text':'Cancel','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_order_active':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_inactive':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_save_ok':{'text':'Place Immediate Orders','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_save_cancel':{'text':'OK','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_order_dialog':{'text':'<b>MOC:</b> Market-On-Close Order. New signals are generated at the close of the market will be placed as Market Orders before the close.<br><b>Immediate:</b> Immediate uses signals generated as of the last Market Close.  If the market is closed, order will be placed as Market-On-Open orders. Otherwise, it will be placed as Market Orders. At the next trigger time, new signals will be placed as MOC orders.','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_save_dialog':{'text':'<center><b>Orders successfully saved.</b><br></center> MOC orders will be placed at the trigger times. If you have entered any immediate orders you may place them now or you may cancel and save different orders.  Any new immediate orders will be placed when the page is refreshed.','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'text_table':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'normal','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_table_title':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_datetimenow':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_triggertimes':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance_account':{'text':'','text-color':'000000','text-font':'Arial Black','text-style':'bold','text-size':'8','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'chip_v4micro':{'text':'5K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_green.png'}},{'chip_v4mini':{'text':'10K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_maroon.png'}},{'chip_v4futures':{'text':'25K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_purple.png'}},]
    #get_blends(cloc=cloc, list_boxstyles=list_boxstyles)

    #for user customized styles (maybe later)
    #list_boxstyles = json.loads(request.GET['boxstyles'])
    #if list_boxstyles != []:
    #   #create new boxstyles json
    #   get_blends(cloc=cloc, list_boxstyles=list_boxstyles)

    votingComponents=get_blends(cloc=cloc, returnVotingComponents=True)

    record = UserSelection(userID=request.GET['user_id'], selection=request.GET['Selection'], \
                            v4futures=json.dumps(votingComponents), v4mini=json.dumps(votingComponents), \
                            v4micro=json.dumps(votingComponents),
                            componentloc = json_cloc,
                            #boxstyles = json_boxstyles,
                            #performance = json_performance,
                            mcdate=MCdate(), timestamp=getTimeStamp(),)
    record.save()
    return HttpResponse(json.dumps({"id": record.id}))


def getrecords(request):
    # records = [ dict((cn, getattr(data, cn)) for cn in ('v4futures', 'v4mini')) for data in UserSelection.objects.all() ]
    # print(records)
    # return HttpResponse(json.dumps(records))
    firstrec = UserSelection.objects.order_by('-timestamp').first()
    if firstrec == None:
        record = UserSelection(userID=json.dumps(UserSelection.default_userid),
                               selection=json.dumps(UserSelection.default_selection),
                               v4futures=json.dumps(UserSelection.default_jsonboard),
                               v4mini=json.dumps(UserSelection.default_jsonboard),
                               v4micro=json.dumps(UserSelection.default_jsonboard),
                               componentloc=json.dumps(UserSelection.default_cloc),
                               mcdate=MCdate(),
                               timestamp=getTimeStamp(), )
        record.save()
        firstrec = UserSelection.objects.order_by('-timestamp').first()

    filename = 'performance_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_performance = json.load(f)
    else:
        list_performance = []
        with open(filename, 'w') as f:
            json.dump(list_performance, f)
        print 'Saved', filename

    filename = 'boxstyles_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_boxstyles = json.load(f)
    else:
        with open(filename, 'w') as f:
            json.dump(UserSelection.default_list_boxstyles, f)
        print 'Saved', filename

    filename = 'customboard_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_customstyles = json.load(f)
    else:
        with open(filename, 'w') as f:
            json.dump(UserSelection.default_list_customboard, f)
        print 'Saved', filename

    firstdata = firstrec.dic()
    firstdata['performance'] = json_performance
    firstdata['boxstyles'] = json_boxstyles
    firstdata['customstyles'] = json_customstyles
    # print(json.dumps(firstdata))
    recent = UserSelection.objects.order_by('-timestamp')[:20]
    recentdata = [dict((cn, getattr(data, cn)) for cn in ('timestamp', 'mcdate', 'selection')) for data in recent]
    returndata={"first": firstdata, "recent": recentdata}
    #print(returndata)
    print len(returndata)
    return HttpResponse(json.dumps(returndata))
    
def index(request):
    return render(request, 'loading_page.html', {})

#def loading_page(request):
#    return render(request, 'loading_page.html', {})

def board(request):
    lastSelection = UserSelection.objects.all().order_by('-timestamp').first()

    #check if any immediate orders
    if lastSelection is not None:
        if 'True' in [order[1] for sys, order in eval(lastSelection.selection).items()]:
            print('processing immediate orders')
            start_immediate()

    '''
    # Please wait 10-15 minutes for the charts to be recreated.
    if selections[0].dic()['componentloc']!=selections[1].dic()['componentloc'] or\
            request.GET['custom_signals']!='False':
        if request.GET['custom_signals']!='False':
            print('New Custom Signals Found')
            recreateCharts(custom_signals=request.GET['custom_signals'])
        else:
            print('New Board config found')
            recreateCharts()
    '''
    return render(request, 'board.html', {})

def newboard(request):
    return render(request, 'newboard.html', {})

def gettimetable(request):
    returndata = get_timetables()
    #print(returndata)
    print len(returndata)
    return HttpResponse(json.dumps(returndata))

def getstatus(request):
    returndata = get_status()
    #print(returndata)
    print len(returndata)
    return HttpResponse(json.dumps(returndata))

def getmetadata(request):
    #returnrec = MetaData.objects.order_by('-timestamp').first()
    #returndata = returnrec.dic()
    returndata=updateMeta()
    #print(returndata)
    print len(returndata)
    return HttpResponse(json.dumps(returndata))

def getaccountdata(request):
    #returnrec = AccountData.objects.order_by('-timestamp').first()
    #returndata = returnrec.dic()
    returndata= getAccountValues()    
    #print(returndata)
    print len(returndata)
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
    files = get_logfiles(search_string='moc_live')
    timestamp = sorted([f[-21:] for f in files])[-1]
    logfile=[f for f in files if 'error' not in f and timestamp in f][-1]
    context['logfiles'].append((logfile,LogFiles(logfile)))

    errorfiles= [f for f in files if logfile[-21:] in f and 'error' in f]
    if len(errorfiles)>0:
        errorfile=errorfiles[-1]
        context['logfiles'].append((errorfile,LogFiles(errorfile)))

    return render(request, 'show_log.html', context)

def errors(request):
    csidate = pd.read_sql('select distinct Date from futuresDF_results',
                          con=getBackendDB()).Date.tolist()[-1]

    files = get_logfiles(search_string='_error_')
    logs=[(logfile,LogFiles(logfile)) for logfile in files if int(logfile.partition('_error_')[-1].split('_')[0]) >= csidate]
    context = {'logfiles': logs}
    return render(request, 'show_errors.html', context)

def logs(request,logfile):
    #files = get_logfiles(search_string='_error_', exclude=True)
    print logfile
    context = {'logfiles': [(logfile,LogFiles('/logs/'+logfile))]}
    return render(request, 'show_logs.html', context)

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