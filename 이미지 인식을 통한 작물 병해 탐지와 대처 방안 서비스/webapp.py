"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""

# https://docs.ultralytics.com/tutorials/pytorch-hub/

# https://github.com/ultralytics/yolov5/issues/36

import argparse
import io
import os
import json
import glob
from PIL import Image
from uuid import uuid4
import torch
from flask import Flask, render_template, request, redirect,url_for
import numpy as np


# 폴더안의 파일 삭제 함수
def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
            
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        DeleteAllFiles('C:/cakd5/strawberry_3rd/strawberry/static/result')  # 탐지 파일 저장소 비우기

        
        # 다중파일 업로드
        if "file" not in request.files:
            return redirect(request.url)
        files = request.files.getlist("file")
        results=[]
        filenames=[]
        if not files:
            return

        pf=[]
        nm=[]
        lk=[]
        st=[]
        for file in files:
            filename = file.filename.rsplit("/")[0]     #파일경로에서 파일명만 추출
            print("processing :", filename)

            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            # print(img)
            # img.save(f"static/bef/{filename}", format="JPEG")
            # print('saving originfile')

            result = model(img, size=1024)
            results.append(result)
            result.render()  # results.imgs에 바운딩박스와 라벨 처리
            data = result.pandas().xyxy[0][['name']].values.tolist()   # results.imgs의 name값만 가져오기
            print("data:",data)

            for img in result.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save(f"static/result/{filename}", format="JPEG")
                print('saving detectfile')

                if len(data) == 0:                          # data 리스트의 값이 0일경우 정상
                    pf.append("정상입니다")
                    nm.append("이상 없음")
                    st.append("뒤로가기를 눌러주세요")
                    lk.append("/") # 작동 안하는 링크
                    
                if len(data) > 0:
                    pf.append("비정상입니다.")# data 리스트의 값이 0이 아닐경우 class name 확인
                    if data == [["Angular Leafspot"]]:            # class name값을 확인하고 번역
                        nm.append("세균모무늬병") 
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=1&sicknsListNo=D00004061")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Anthracnose Fruit Rot"]]:
                        nm.append("탄저병")
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=2&sicknsListNo=D00000448")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Blossom Blight"]]:
                        nm.append("꽃 곰팡이병")
                        lk.append("http://www.nongsaro.go.kr/portal/ps/psb/psbb/farmUseTechDtl.ps;jsessionid=CqT32sPeKoNArbirTrhdvQ6kD7uu7OZSky1xnNjWkWL2vk3KMZvN1sFz3Pl0u65m.nongsaro-web_servlet_engine1?menuId=PS00072&farmPrcuseSeqNo=15659&totalSearchYn=Y")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Gray Mold"]]:
                        nm.append("잿빛 곰팡이병")
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=2&sicknsListNo=D00000440")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Leaf Spot"]]:
                        nm.append("뱀눈무늬병")
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=1&sicknsListNo=D00000449")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Powdery Mildew Fruit"]]:
                        nm.append("흰 가루병(과실)")
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=2&sicknsListNo=D00000459")
                        st.append(" 원인/대처법 보러 가기 ")
                    if data == [["Powdery Mildew Leaf"]]:
                        nm.append("흰 가루병(잎)")
                        lk.append("https://ncpms.rda.go.kr/npms/SicknsInfoDtlR.np?sSearchWord=%EB%94%B8%EA%B8%B0&sKncrCode01=&sKncrCode02=&sKncrCode=&sch2=&sch3=&sSearchOpt=&pageIndex=2&sicknsListNo=D00000459")
                        st.append(" 원인/대처법 보러 가기 ")
            
            
            root = "static/result/"
            filenames.append(root + filename)
            print(pf)
            print(nm)
            print(lk)
            
        return render_template("result.html",files=filenames,pf=pf,nm=nm,lk=lk,st=st,enumerate=enumerate,len=len)
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

# local model

    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', 'C:\cakd5\strawberry_3rd\strawberry\model.pt', autoshape=True
    )  # force_reload = recache latest code
    model.eval()

    flask_options = dict(
        host='0.0.0.0',
        debug=True,
        port=args.port,
        threaded=True,
    )

    app.run(**flask_options)

        



