from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Dictionary
from .forms import TreasureForm
# Create your views here.
'''
class Treasure:
    def __init__(self, name, value, material, location):
        self.name = name
        self.value = value
        self.material = material
        self.location = location

treasures = [
    Treasure('Gold Nugget', 500.00, 'gold', "Curly's Creek, NM"),
    Treasure("Fool's Gold", 0, 'pyrite', "Fool's Falls, CO"),
    Treasure('Coffee Can', 20.0, 'tin', 'Acme, CA')
    ]
'''
    

def index(request):
    #return HttpResponse('<h1>Hello Explorers!</h1>')
    #name = 'Gold Nugget'
    #value = 1000.00
    #context = {'treasure_name': name, 'treasure_val' : value}
    form = TreasureForm()
    treasures=Dictionary.objects.all()
    context ={'treasures':treasures, 'form':form}
    return render(request, 'index.html', context)
    
def detail(request, treasure_id):
    treasure = Dictionary.objects.get(id=treasure_id)
    context ={'treasure':treasure}
    return render(request, 'detail.html', context)

def post_treasure(request):
    form = TreasureForm(request.POST)
    if form.is_valid():
        treasure = Treasure(
                            name = form.cleaned_data['name'],
                            value = form.cleaned_data['value'],
                            material = form.cleaned_data['material'],
                            location = form.cleaned_data['location'],
                            img_url = form.cleaned_data['img_url']
                            )
        treasure.save()
    return HttpResponseRedirect('/')