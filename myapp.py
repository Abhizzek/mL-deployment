from flask import Flask,render_template,request
from model import *

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/getpred",methods=['GET','POST'])
def getpredict():
    if request.method=='POST':
        ae=request.form['ae']
        se=request.form['se']
        st=request.form['st']
        av=request.form['av']
        fg=request.form['fg']
        ml=request.form['ml']
        aa=request.form['aa']
        lb=request.form['lb']
        lf=request.form['lf']
        sp=request.form['sp']
        ss=request.form['ss']
        at=request.form['at']
        vc=request.form['vc']
        br=request.form['br']
        ap=request.form['ap']
        sg=request.form['sg']
        an=request.form['an']
        pe=request.form['pe']
        hy=request.form['hy']

        newobs=np.array([[ae,se,st,av,fg,ml,aa,lb,lf,sp,ss,at,vc,br,ap,sg,an,pe,hy]],dtype=int)
        print(newobs)
        model=makepredict()
        yp=model.predict(newobs)[0]
        print(yp)
        return render_template("index1.html",data=yp)
    
if __name__=="__main__":
    app.run(debug=True)
