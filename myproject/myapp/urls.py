# -*- coding: utf-8 -*-
from django.conf.urls import url
from myproject.myapp.views import list
from myproject.myapp.views import result
<<<<<<< HEAD
=======

>>>>>>> 9c234430350850fde7739430f5b55825302ddc9a

urlpatterns = [
    url(r'^list/$', list, name='list'),
    url(r'^result/$', result, name='result')
]
