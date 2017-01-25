from django.db import models
import json
from os.path import isfile, join

class UserSelection(models.Model):
    #defaults
    default_userid=32
    default_cloc=[{'c0': 'Off'}, {'c1': 'RiskOn'}, {'c2': 'RiskOff'}, {'c3': 'LowestEquity'}, {'c4': 'HighestEquity'},
                {'c5': 'AntiHighestEquity'}, {'c6': 'Anti50/50'}, {'c7': 'Seasonality'}, {'c8': 'Anti-Seasonality'},
                {'c9': 'Previous'}, {'c10': 'None'}, {'c11': 'Anti-Previous'}, {'c12': 'None'}, {'c13': 'None'},
                {'c14': 'None'}, ]
    json_cloc = json.dumps(default_cloc)
    default_selection={'v4futures': ['Off', 'False'], 'v4micro': ['Off', 'False'], 'v4mini': ['Off', 'False']}
    default_board={"1":["0.5LastSIG","RiskOn"],"2":["AntiPrevACT","0.5LastSIG","RiskOn"],"3":["prevACT","0.5LastSIG","RiskOn"],"4":["0.5LastSIG","RiskOff"],"5":["AntiPrevACT","0.5LastSIG","RiskOff"],"6":["prevACT","0.5LastSIG","RiskOff"],"7":["1LastSIG","RiskOn"],"8":["AntiPrevACT","1LastSIG","RiskOn"],"9":["prevACT","1LastSIG","RiskOn"],"10":["1LastSIG","RiskOff"],"11":["AntiPrevACT","1LastSIG","RiskOff"],"12":["prevACT","1LastSIG","RiskOff"],"13":["Anti1LastSIG","RiskOn"],"14":["AntiPrevACT","Anti1LastSIG","RiskOn"],"15":["prevACT","Anti1LastSIG","RiskOn"],"16":["Anti1LastSIG","RiskOff"],"17":["AntiPrevACT","Anti1LastSIG","RiskOff"],"18":["prevACT","Anti1LastSIG","RiskOff"],"19":["Anti0.75LastSIG","RiskOn"],"20":["AntiPrevACT","Anti0.75LastSIG","RiskOn"],"21":["prevACT","Anti0.75LastSIG","RiskOn"],"22":["Anti0.75LastSIG","RiskOff"],"23":["AntiPrevACT","Anti0.75LastSIG","RiskOff"],"24":["prevACT","Anti0.75LastSIG","RiskOff"],"25":["LastSEA","RiskOn"],"26":["AntiPrevACT","LastSEA","RiskOn"],"27":["prevACT","LastSEA","RiskOn"],"28":["LastSEA","RiskOff"],"29":["AntiPrevACT","LastSEA","RiskOff"],"30":["prevACT","LastSEA","RiskOff"],"31":["AntiSEA","RiskOn"],"32":["AntiPrevACT","AntiSEA","RiskOn"],"33":["prevACT","AntiSEA","RiskOn"],"34":["AntiSEA","RiskOff"],"35":["AntiPrevACT","AntiSEA","RiskOff"],"36":["prevACT","AntiSEA","RiskOff"],"Off":["None"],"RiskOn":["RiskOn"],"RiskOff":["RiskOff"],"LowestEquity":["0.5LastSIG"],"HighestEquity":["1LastSIG"],"AntiHighestEquity":["Anti1LastSIG"],"Anti50/50":["Anti0.75LastSIG"],"Seasonality":["LastSEA"],"Anti-Seasonality":["AntiSEA"],"Previous":["prevACT"],"Anti-Previous":["AntiPrevACT"],}
    default_list_boxstyles = [{'c0':{'text':'Off','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'225823','fill-R':'34','fill-G':'88','fill-B':'35','filename':''}},{'c1':{'text':'RiskOn','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'BE0032','fill-R':'190','fill-G':'0','fill-B':'50','filename':''}},{'c2':{'text':'RiskOff','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'222222','fill-R':'34','fill-G':'34','fill-B':'34','filename':''}},{'c3':{'text':'LowestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F38400','fill-R':'243','fill-G':'132','fill-B':'0','filename':''}},{'c4':{'text':'HighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'FFFF00','fill-R':'255','fill-G':'255','fill-B':'0','filename':''}},{'c5':{'text':'AntiHighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'A1CAF1','fill-R':'161','fill-G':'202','fill-B':'241','filename':''}},{'c6':{'text':'Anti50/50','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'C2B280','fill-R':'194','fill-G':'178','fill-B':'128','filename':''}},{'c7':{'text':'Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'E68FAC','fill-R':'230','fill-G':'143','fill-B':'172','filename':''}},{'c8':{'text':'Anti-Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F99379','fill-R':'249','fill-G':'147','fill-B':'121','filename':''}},{'c9':{'text':'Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'654522','fill-R':'101','fill-G':'69','fill-B':'34','filename':''}},{'c10':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c11':{'text':'Anti-Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'008856','fill-R':'0','fill-G':'136','fill-B':'86','filename':''}},{'c12':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c13':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c14':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_clear_all':{'text':'Clear All Bets','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_create_new':{'text':'Create New Board','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_confirm_orders':{'text':'Save Orders','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_ok':{'text':'Enter Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_order_cancel':{'text':'Cancel','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_order_active':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_inactive':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_save_ok':{'text':'Place Immediate Orders Now','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_save_cancel':{'text':'OK/Change Immediate Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_order_dialog':{'text':'<b>MOC:</b> Market-On-Close Order. New signals are generated at the close of the market will be placed as Market Orders before the close.<br><b>Immediate:</b> Immediate uses signals generated as of the last Market Close.  If the market is closed, order will be placed as Market-On-Open orders. Otherwise, it will be placed as Market Orders. At the next trigger time, new signals will be placed as MOC orders.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'d_save_dialog':{'text':'<center><b>Orders successfully saved.</b><br></center> MOC orders will be placed at the trigger times. If you have entered any immediate orders you may place them now or you may cancel and save different orders.  After the page is refreshed you can check order status to see if the orders were placed.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'text_table':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'16','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_table_title':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_datetimenow':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_triggertimes':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance_account':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'chip_v4micro':{'text':'50K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_maroon.png'}},{'chip_v4mini':{'text':'100K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_purple.png'}},{'chip_v4futures':{'text':'250K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_orange.png'}},]
    default_list_customboard=[{'background':{'text':'','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c0':{'text':'Off','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'cNone':{'text':'','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_auto_select':{'text':'Auto-Select','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_reset_colors':{'text':'Reset','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_save_colors':{'text':'Save Colors','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_reset_board':{'text':'Reset','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'FFFFFF','fill-R':'255','fill-G':'255','fill-B':'255','filename':''}},{'b_save_board':{'text':'Save Colors','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'text_components':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_choose_colors':{'text':'<b>Step 1</b> <br>Click on the component box to choose the colors for the components you want to use. Once you are done click the save button. If you do not want to choose custom colors, click Auto-Select and save.','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_place_components':{'text':'<b>Step 2</b> <br>Drag and drop components to blank boxes below. You may leave boxes blank.','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'d_save_color_error':{'text':'Please color "Off" component and at least one other component.','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'d_save_board_error':{'text':'Please place at least one component in one of the blank boxes.','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'list_autoselect':[{'fill-colorname':'red','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'BE0032','fill-R':'190','fill-G':'0','fill-B':'50'},{'fill-colorname':'black','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'222222','fill-R':'34','fill-G':'34','fill-B':'34'},{'fill-colorname':'lime','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'4FF773','fill-R':'79','fill-G':'247','fill-B':'115'},{'fill-colorname':'yellow','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'FFFF00','fill-R':'255','fill-G':'255','fill-B':'0'},{'fill-colorname':'lightblue','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'A1CAF1','fill-R':'161','fill-G':'202','fill-B':'241'},{'fill-colorname':'buff','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'C2B280','fill-R':'194','fill-G':'178','fill-B':'128'},{'fill-colorname':'purplishpink','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'E68FAC','fill-R':'230','fill-G':'143','fill-B':'172'},{'fill-colorname':'yellowishpink','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'F99379','fill-R':'249','fill-G':'147','fill-B':'121'},{'fill-colorname':'orange','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'F38400','fill-R':'243','fill-G':'132','fill-B':'0'},{'fill-colorname':'grey','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'848482','fill-R':'132','fill-G':'132','fill-B':'130'},{'fill-colorname':'green','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'008856','fill-R':'0','fill-G':'136','fill-B':'86'},{'fill-colorname':'blue','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'0067A5','fill-R':'0','fill-G':'103','fill-B':'165'},{'fill-colorname':'violet','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'604E97','fill-R':'96','fill-G':'78','fill-B':'151'},{'fill-colorname':'purplishred','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'B3446C','fill-R':'179','fill-G':'68','fill-B':'108'},{'fill-colorname':'yellowishbrown','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'654522','fill-R':'101','fill-G':'69','fill-B':'34'},{'fill-colorname':'reddishorange','text-color':'','text-font':'','text-style':'','text-size':'','fill-Hex':'EA2819','fill-R':'235','fill-G':'40','fill-B':'25'},]},]

    filename='performance_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_performance = json.load(f)
    else:
        list_performance=[]
        with open(filename, 'w') as f:
            json.dump(list_performance, f)
        print 'Saved', filename

    filename='boxstyles_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_boxstyles = json.load(f)
    else:
        with open(filename, 'w') as f:
            json.dump(default_list_boxstyles, f)
        print 'Saved', filename
        
    filename='customboard_data.json'
    if isfile(filename):
        with open(filename, 'r') as f:
            json_customstyles = json.load(f)
    else:
        with open(filename, 'w') as f:
            json.dump(default_list_customboard, f)
        print 'Saved', filename

    userID = models.IntegerField()
    selection = models.TextField()
    v4futures = models.TextField()
    v4mini = models.TextField()
    v4micro = models.TextField()
    componentloc = models.TextField(default=json_cloc)
    #boxstyles = models.TextField(default=json_boxstyles)
    #performance = models.TextField(default=json_performance)
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def dic(self):
        #fields = ['selection', 'v4futures', 'v4mini', 'v4micro', 'componentloc','boxstyles','performance', 'mcdate', 'timestamp']
        fields = ['selection', 'v4futures', 'v4mini', 'v4micro', 'componentloc', 'mcdate',
                  'timestamp']
        result = {}
        for field in fields :
            result[field] = self.__dict__[field]
        result['performance'] = self.json_performance
        result['boxstyles'] = self.json_boxstyles
        result['customstyles'] = self.json_customstyles
        return result

    def __str__(self):
        return self.selection

class MetaData(models.Model):
    components = models.TextField()
    triggers = models.TextField()
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def dic(self):
        fields = ['components', 'triggers', 'mcdate', 'timestamp']
        result = {}
        for field in fields :
            result[field] = self.__dict__[field]
        return result

    def __str__(self):
        return self.mcdate


class AccountData(models.Model):
    value1 = models.TextField()
    value2 = models.TextField()
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def dic(self):
        fields = ['value1', 'value2', 'mcdate', 'timestamp']
        result = {}
        for field in fields :
            result[field] = self.__dict__[field]
        return result

    def __str__(self):
        return self.mcdate