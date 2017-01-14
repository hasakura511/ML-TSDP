from django.db import models

class UserSelection(models.Model):
    userID = models.IntegerField()
    selection = models.TextField()
    v4futures = models.TextField()
    v4mini = models.TextField()
    v4micro = models.TextField()
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def dic(self):
        fields = ['selection', 'v4futures', 'v4mini', 'v4micro', 'mcdate', 'timestamp']
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