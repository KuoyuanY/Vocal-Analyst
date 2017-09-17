# -*- coding: utf-8 -*-
from django.db import models
import os
import uuid
from django.dispatch import receiver
from django.utils.translation import ugettext_lazy as _

class Document(models.Model):
    docfile = models.FileField(upload_to='documents/%Y/%m/%d')
