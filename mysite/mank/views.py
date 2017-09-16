# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

# Create your views here.

def index(request):
    return render(request, 'mank/index.html')

def simple_upload(request):
    if request.method == 'POST' and request.FILES['first'] and request.FILES['second']:
        fs = FileSystemStorage()
        firstname = fs.save(first.name, first)
        secondname = fs.save(second.name, second)
        uploaded_file_url = fs.url(firstname)
        return render(request, 'mank/index.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'mank/index.html')

#still need to create result.html
def analysis(request):
    return render(request, 'mank/result.html', {
        'score': score 
    })
