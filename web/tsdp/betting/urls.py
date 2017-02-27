from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^addrecord$', views.addrecord),
    url(r'^getrecords$', views.getrecords),
    url(r'^getmetadata$', views.getmetadata),
    url(r'^getaccountdata$', views.getaccountdata),
    url(r'^$', views.index, name='index'),
    url(r'^login/$', views.login_view, name='login'),
    url(r'^logout/$', views.logout_view, name='logout'),
    url(r'^register/$', views.register, name='register'),
    url(r'^user/(\w+)/$', views.profile, name='profile'),
	url(r'^last_userselection/$', views.last_userselection, name='last_userselection'),
    url(r'^downloaddb/$', views.downloaddb, name='downloaddb'),
    url(r'^futures/$', views.futures, name='futures'),
    url(r'^symbols/$', views.symbols, name='symbols'),
    url(r'^([A-Z\d]+)/$', views.system_charts, name='system_charts'),
    url(r'^order_status/$', views.order_status, name='order_status'),
    url(r'^board/$', views.board, name='board'),
    url(r'^timetable/$', views.timetable, name='timetable'),
    url(r'^archive/$', views.archive, name='archive'),
    url(r'^archive/(?P<date>[0-9-]+)/$', views.futures, name='futures'),
    url(r'^logs/(.*)$', views.logs, name='logs'),
    url(r'^errors/$', views.errors, name='errors'),
    url(r'^gettimetable$', views.gettimetable),
    url(r'^getstatus', views.getstatus),
    url(r'^newboard/$', views.newboard),
]
