import cv2,pandas,time
from datetime import datetime
from bokeh.plotting import figure
from bokeh.io import output_file,show
from bokeh.models import HoverTool,ColumnDataSource

status_list=[None,None]
time_list=[]
video=cv2.VideoCapture(0)
first_frame=None
time.sleep(5)
df=pandas.DataFrame(columns=["start","end"])

while True:
    check, frame=video.read()
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    cv2.imshow("frame",gray)
    if first_frame is None:
        first_frame=gray
        continue

    diff=cv2.absdiff(gray,first_frame)
    thresh_frame=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame,None,iterations=2)

    (_,cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for countour in cnts:
        if  cv2.contourArea(countour)<10000:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(countour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(128,0,0),3)


    status_list.append(status)
    status_list=status_list[-2:]

    if(status_list[-1]==1 and status_list[-2]==0):
        time_list.append(datetime.now())
        print("appended")
    if(status_list[-1]==0 and status_list[-2]==1):
        time_list.append(datetime.now())
        print("appended")


    cv2.imshow("thresh",thresh_frame)
    cv2.imshow("frame",frame)
    #print(status)

    print(status_list)
    key=cv2.waitKey(1)

    if(key==ord('q')):
        if status==1:
            time_list.append(datetime.now())
            print("appended")
        break


for i in range(0,len(time_list)-1,2):
    df=df.append({"start":time_list[i],"end":time_list[i+1]},ignore_index=True)

print(time_list)
df.to_csv("data.csv")
video.release()
cv2.destroyAllWindows()
df=pandas.read_csv("data.csv",parse_dates=["start","end"])
f=figure(width=800,height=200,x_axis_type="datetime",title="Motion Graph")
f.yaxis.minor_tick_line_color = None
f.ygrid[0].ticker.desired_num_ticks=1
df["start_string"]=df["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["end_string"]=df["end"].dt.strftime("%Y-%m-%d %H:%M:%S")
hover=HoverTool(tooltips=[("start","@start_string"),("end","@end_string")])
f.add_tools(hover)
cds=ColumnDataSource(df)
q=f.quad(left="start",right="end",bottom=0,top=1,color="green",source=cds)
output_file("output.html")
show(f)
