from django.db import models

class UserSelection(models.Model):
    userID = models.IntegerField()
    selection = models.TextField()
    v4futures = models.TextField()
    v4mini = models.TextField()
    v4micro = models.TextField()
    mcdate = models.TextField()
    timestamp = models.IntegerField()

    def __str__(self):
        return self.selection