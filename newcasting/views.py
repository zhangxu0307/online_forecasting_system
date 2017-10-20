#encoding=utf-8
from django.shortcuts import render
from django.http import HttpResponse
import os
import pandas as pd
import data_analysis
import json
from .forms import ArguForm

# Create your views here.


def home(request):
    return render(request, 'info.html')

def showDoc(request):
    return render(request,'doc.html')

def uploadFile(request):

    ans = None
    dataDcit = None
    evalans = None
    accMean = 0
    accList = None
    BASE_DIR = os.path.dirname(__file__)
    a = b = 12

    if request.method == "POST":
        myFile =request.FILES.get("myfile", None)  # 获取文件
        if not myFile:  
            return HttpResponse("no files for upload!")
        filePath = BASE_DIR+"/data/data1.csv"
        destination = open(filePath,'wb+')  # 建立目标文件
        for chunk in myFile.chunks():    # 分块写入
            destination.write(chunk)  
        destination.close()

        form = ArguForm(request.POST)  # form 包含提交的数据
        if form.is_valid():  # 如果提交的数据合法
            a = form.cleaned_data['windowSize']
            b = form.cleaned_data['foreHead']
            if a < 0 or b < 0 or b > 12:
                return HttpResponse("Invalid Argument!")

        df = pd.read_csv(BASE_DIR + '/data/data1.csv', encoding='utf-8', index_col='date', skip_footer=0)
        #dataDcit = df.to_dict() # 按年份形成字典
        df.index = pd.to_datetime(df.index)
        dataDcit = data_analysis.castingData(df) # 按月份形成字典
        dataset = df['x'].values
        if dataset.shape[0] == 0:
            return HttpResponse("No Valid Data!")
        dataset = dataset.astype('float64')

        x, y = data_analysis.create_dataset(dataset,a,b)
        ans = data_analysis.newcasting(x, y,a,b)

        evalans,accList,accMean = data_analysis.evaluate2(dataset,a,b)

    else:  # 当正常访问时
        form = ArguForm()

    return render(request, 'charts.html', {'List': json.dumps(ans),
                                               'Dict': json.dumps(dataDcit),'List2':json.dumps(evalans),
                                           'List3':json.dumps([accMean]),'List4':json.dumps(accList),'form':form})

    # return render(request, 'success.html')

# def getReport(request):
#
#     BASE_DIR = os.path.dirname(__file__)
#     df = pd.read_csv(BASE_DIR + '/data/data1.csv', encoding='utf-8', index_col='date', skip_footer=0)
#     df.index = pd.to_datetime(df.index)
#     dataDcit = data_analysis.castingData(df)
#     dataset = df['x'].values
#     if dataset.shape[0] == 0:
#         return HttpResponse("No Valid Data!")
#     dataset = dataset.astype('float64')
#
#     x, y = data_analysis.create_dataset(dataset)
#     ans = data_analysis.newcasting(x, y)
#     return render(request, 'report.html', {'List': json.dumps(ans),
#                                            'Dict': json.dumps(dataDcit)})











