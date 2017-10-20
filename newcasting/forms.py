#encoding=utf-8
from django import forms

class ArguForm(forms.Form):
    windowSize = forms.IntegerField(label=u"请选择预测窗口大小：")
    foreHead = forms.IntegerField(label=u"请选择向前预测数据个数：")