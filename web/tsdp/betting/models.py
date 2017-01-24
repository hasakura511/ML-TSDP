from django.db import models
import json


class UserSelection(models.Model):
    #cloc, list_boxstyles = get_blends()
    #json_boxstyles = json.dumps(list_boxstyles)
    cloc=[{'c0': 'Off'}, {'c1': 'RiskOn'}, {'c2': 'RiskOff'}, {'c3': 'LowestEquity'}, {'c4': 'HighestEquity'},
                {'c5': 'AntiHighestEquity'}, {'c6': 'Anti50/50'}, {'c7': 'Seasonality'}, {'c8': 'Anti-Seasonality'},
                {'c9': 'Previous'}, {'c10': 'None'}, {'c11': 'Anti-Previous'}, {'c12': 'None'}, {'c13': 'None'},
                {'c14': 'None'}, ]
    json_cloc = json.dumps(cloc)

    with open('performance_data.json', 'r') as f:
        json_performance = json.load(f)
    with open('boxstyles_data.json', 'r') as f:
        json_boxstyles = json.load(f)

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