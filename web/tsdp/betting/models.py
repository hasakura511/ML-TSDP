from django.db import models

class UserSelection(models.Model):

    def get_blends(cloc=None, list_boxstyles=None):
        def is_int(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        if cloc==None:
            cloc = [{'c0':'Off'},{'c1':'RiskOn'},{'c2':'RiskOff'},{'c3':'LowestEquity'},{'c4':'HighestEquity'},{'c5':'AntiHighestEquity'},{'c6':'Anti50/50'},{'c7':'Seasonality'},{'c8':'Anti-Seasonality'},{'c9':'Previous'},{'c10':'None'},{'c11':'Anti_previous'},{'c12':'None'},{'c13':'None'},{'c14':'None'},]


        if list_boxstyles==None:
            list_boxstyles = [{'c0':{'text':'Off','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'225823','fill-R':'34','fill-G':'88','fill-B':'35','filename':''}},{'c1':{'text':'RiskOn','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'BE0032','fill-R':'190','fill-G':'0','fill-B':'50','filename':''}},{'c2':{'text':'RiskOff','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'222222','fill-R':'34','fill-G':'34','fill-B':'34','filename':''}},{'c3':{'text':'LowestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F38400','fill-R':'243','fill-G':'132','fill-B':'0','filename':''}},{'c4':{'text':'HighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'FFFF00','fill-R':'255','fill-G':'255','fill-B':'0','filename':''}},{'c5':{'text':'Anti-HighestEquity','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'A1CAF1','fill-R':'161','fill-G':'202','fill-B':'241','filename':''}},{'c6':{'text':'Anti-50/50','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'C2B280','fill-R':'194','fill-G':'178','fill-B':'128','filename':''}},{'c7':{'text':'Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'E68FAC','fill-R':'230','fill-G':'143','fill-B':'172','filename':''}},{'c8':{'text':'Anti-Seasonality','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F99379','fill-R':'249','fill-G':'147','fill-B':'121','filename':''}},{'c9':{'text':'Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'654522','fill-R':'101','fill-G':'69','fill-B':'34','filename':''}},{'c10':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c11':{'text':'Anti-Previous','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'008856','fill-R':'0','fill-G':'136','fill-B':'86','filename':''}},{'c12':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c13':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'c14':{'text':'None','text-color':'FFFFFF','text-font':'Book Antigua','text-style':'bold','text-size':'24','fill-Hex':'F2F3F4','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_clear_all':{'text':'Clear All Bets','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_create_new':{'text':'Create New Board','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_confirm_orders':{'text':'Save Orders for Processing','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'14','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_ok':{'text':'Enter Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_order_cancel':{'text':'Cancel','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_order_active':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'33CC00','fill-R':'51','fill-G':'204','fill-B':'0','filename':''}},{'b_order_inactive':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'b_save_ok':{'text':'Place Immediate Orders Now','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'29ABE2','fill-R':'41','fill-G':'171','fill-B':'226','filename':''}},{'b_save_cancel':{'text':'OK/Change Immediate Orders','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'d_order_dialog':{'text':'<b>MOC:</b> Market-On-Close Order. New signals are generated at the close of the market will be placed as Market Orders before the close.<br><b>Immediate:</b> Immediate uses signals generated as of the last Market Close.  If the market is closed, order will be placed as Market-On-Open orders. Otherwise, it will be placed as Market Orders. At the next trigger time, new signals will be placed as MOC orders.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'d_save_dialog':{'text':'<center><b>Orders successfully saved.</b><br></center> MOC orders will be placed at the trigger times. If you have entered any immediate orders you may place them now or you may cancel and save different orders.  Any new immediate orders will be placed when the page is refreshed.','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'14','fill-Hex':'FFFFFF','fill-R':'242','fill-G':'243','fill-B':'244','filename':''}},{'text_table':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'normal','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_table_title':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_datetimenow':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_triggertimes':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'text_performance_account':{'text':'','text-color':'000000','text-font':'Book Antigua','text-style':'bold','text-size':'18','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':''}},{'chip_v4micro':{'text':'50K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_maroon.png'}},{'chip_v4mini':{'text':'100K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_purple.png'}},{'chip_v4futures':{'text':'250K','text-color':'000000','text-font':'','text-style':'','text-size':'','fill-Hex':'','fill-R':'','fill-G':'','fill-B':'','filename':'chip_orange.png'}},]
        else:
            list_boxstyles = [d for d in list_boxstyles if not is_int(d.keys())[0]]

        #print([cl.keys()[0] for cl in cloc])
        component_styles = {bs.keys()[0]: bs.values()[0] for bs in list_boxstyles if
                            bs.keys()[0] in [cl.keys()[0] for cl in cloc]}
        component_names = {cl.keys()[0]: cl.values()[0] for cl in cloc}

        # get from board js
        h_components = 6
        v_components = 6
        v_components_width = 2
        h_components_width = outside_components = 2
        table_height = 3

        table_width = num_v_component_boxes = h_components * outside_components
        num_h_component_boxes = outside_components * table_height
        total_boxes = table_height * table_width
        start_vert = outside_components + h_components
        end_vert = outside_components + h_components + v_components

        # figure this out later..
        # mod:boxkeys
        # vboxdict={0:['c9','c10'].
        #                2:['c11','c12'],
        #                1:['c13','c14'],
        #            }
        vboxdict = {}
        vboxlist = [x for x in range(table_height - 1, 0, -1)]
        vboxlist.insert(0, 0)
        for x in vboxlist:
            vboxdict[x] = []
            for y in range(0, v_components_width):
                start_vert += 1
                # print x,y,start_vert
                vboxdict[x].append('c' + str(start_vert))

        # 1-6 +c3 ... 31-36 +c8
        boxidDict = {}
        for boxid in range(1, total_boxes + 1):
            # print
            h_component = int(math.ceil(boxid / float(num_h_component_boxes)))
            boxidDict[str(boxid)] = ['c' + str(h_component + outside_components)]
            o_component = int(math.ceil(boxid / float(table_height))) - outside_components * (h_component - 1)
            boxidDict[str(boxid)] += ['c' + str(o_component)]
            boxidDict[str(boxid)] += vboxdict[boxid % table_height]

        boxstyleDict = {boxid: [component_styles[x] for x in boxidDict[boxid] if component_names[x] is not 'None'] for
                        boxid
                        in boxidDict}

        blendedboxstyleDict = {}
        for boxid, list_of_styles in boxstyleDict.items():
            # fillhex_test={}
            R = 0
            G = 0
            B = 0
            for i, style in enumerate(list_of_styles):
                blendedstyle = style.copy()
                # fillhex_test[i]=('%02x%02x%02x' % (int(style['fill-R']), int(style['fill-G']), int(style['fill-B']))).upper()
                R += int(blendedstyle['fill-R'])
                G += int(blendedstyle['fill-G'])
                B += int(blendedstyle['fill-B'])
                # print i, blendedstyle, R, G, B
            i += 1
            BR = int(R / float(i))
            BG = int(G / float(i))
            BB = int(B / float(i))
            fillhex = ('%02x%02x%02x' % (BR, BG, BB)).upper()
            # print i, BR, BG, BB, fillhex
            # blended = blend_colors(list_of_blendedstyles)
            L = 0
            ldict = {'r': 0.2126, 'g': 0.7152, 'b': 0.0722}
            for color, value in [('r', BR), ('g', BG), ('b', BB)]:
                c = value / 255.0
                if c <= 0.03928:
                    c = c / 12.92
                else:
                    c = ((c + 0.055) / 1.055) ** 2.4
                L += c * ldict[color]
            textcolor = '000000' if L > 0.179 else 'FFFFFF'
            # print L, textcolor

            blendedstyle.update({
                'fill-R': str(BR),
                'text-color': textcolor,
                'fill-Hex': fillhex,
                'fill-G': str(BG),
                'fill-B': str(BB),
                'text': str(boxid),
                # 'text-size': '24',
                # 'text-blendedstyle': 'bold',
                # 'text-font': 'Book Antigua',
                #'fill-colorname': 'blended'
            })
            blendedboxstyleDict[boxid] = blendedstyle
            # print boxid, style
        keys = blendedboxstyleDict.keys()
        keys.sort(key=int)
        list_boxstyles+=[{key: blendedboxstyleDict[key]} for key in keys]
        return cloc, list_boxstyles

    cloc, list_boxstyles = get_blends()
    json_boxstyles = json.dumps(list_boxstyles)
    json_cloc = json.dumps(cloc)

    with open('performance_data.json', 'r') as f:
        json_performance = json.load(f)

    userID = models.IntegerField()
    selection = models.TextField()
    v4futures = models.TextField()
    v4mini = models.TextField()
    v4micro = models.TextField()
    componentloc = models.TextField(default=json_cloc)
    boxstyles = models.TextField(default=json_boxstyles)
    performance = models.TextField(default=json_performance)
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def dic(self):
        fields = ['selection', 'v4futures', 'v4mini', 'v4micro', 'componentloc','boxstyles','performance', 'mcdate', 'timestamp']
        result = {}
        for field in fields :
            result[field] = self.__dict__[field]
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