# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.core.urlresolvers import reverse
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=120)
    image = models.FileField(null=True, blank=True)
    content = models.TextField()
    updated = models.DateTimeField(auto_now=True, auto_now_add=False)
    timestamp = models.DateTimeField(auto_now=False, auto_now_add=True)

    def _unicode_(self):
        return self.title

    def _str_(self):
        return self.title

    def get_absolute_url(self):
        return reverse("post:detail", kwargs={"id": self.id})

    class Meta:
        ordering = ["-timestamp", "-updated"]
    
# Create your models here.
