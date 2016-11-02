from models import Treasure

t = Treasure('Gold Nugget2', 5000.00, 'gold', "Curly's Creek, NM")
t.save()
t = Treasure("Fool's Gold2", 00, 'pyrite', "Fool's Falls, CO")
t.save()
t=Treasure('Coffee Can2', 200.0, 'tin', 'Acme, CA')
t.save()

with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            _, created = Teacher.objects.get_or_create(
                first_name=row[0],
                last_name=row[1],
                middle_name=row[2],
                )
            # creates a tuple of the new object or
            # current object and a boolean of if it was created
            
# open file & create csvreader
import csv, yada yada yada

# import the relevant model
from myproject.models import Foo

#loop:
for line in csv file:
     line = parse line to a list
     # add some custom validation\parsing for some of the fields

     foo = Foo(fieldname1=line[1], fieldname2=line[2] ... etc. )
     try:
         foo.save()
     except:
         # if the're a problem anywhere, you wanna know about it
         print "there was a problem with line", i 