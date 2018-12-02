import pandas as pd
import numpy as np
import math
import tensorflow as tf
import copy

#===========================================
# 1. read csv
#===========================================
#대상 정격용량 년 월 일 시 인버터출력 경사일사량 수평일사량 표면온도 주위온도
fullData = pd.read_csv("./data/pv_2015_2016.csv", encoding='utf-8')
fullData_original = np.array(fullData)
fullData_arrange = np.array(fullData)[:,[2,3,4,5,6,7,8,9,10,1]] #년,월,일,시,태양광발전량,경사일사량,수평일사량,표면온도,주위온도,정격전압


#=========================================
# 2. find group start index and split
#=========================================
temp = ""
for e,d in enumerate(fullData_original):
    if d[0] != temp:
        print(e,d[0])
        temp = d[0]

gy = fullData_arrange[0:17520]
sp = fullData_arrange[17520:33958]
ya = fullData_arrange[33958:]

print(gy[-1], sp[-1], ya[-1])
print("split len:", len(gy), len(sp), len(ya))

#=========================================
# 3. refine each group
#=========================================
def refine_missing(data):
    refined =np.array(data[0]).reshape(-1,len(data[0]))
    i = 1
    while (i < len(data)):
        timeDif = int(data[i, 3]) - int(refined[-1, 3])
        if (timeDif == 1 or timeDif == -23):
            refined = np.append(refined, data[i].reshape(-1, len(data[i])), axis=0)
        else:
            temp = refined[-1]
            temp[3] = (temp[3] + 1) % 24
            temp[4:-1] = 0
            print("generated!:",temp, i)
            refined = np.append(refined, temp.reshape(-1, len(data[i])), axis=0)
            i -= 1
        i += 1
    return refined

def refine_overmax(data):
    refined = copy.deepcopy(data)
    for d in refined:
        if(d[4]> d[9]):
            d[4] = d[9]
    return refined

def refine(data):
    refined = refine_missing(data)
    refined = refine_overmax(refined)
    return refined

gy_refined = refine(gy)
sp_refined = refine(sp)
ya_refined = refine(ya)
print("refined len:",len(gy_refined),len(sp_refined),len(ya_refined))


#=========================================
# 3. refine each group
#=========================================
saver_gy = pd.DataFrame(gy_refined,columns=["year","month","day","hour","gen","S_irradiation","H_irradiation","surface_temp","surround_temp","maxgen"])
saver_gy.to_csv("./data/pv_2015_2016_gy_processed.csv",index=False)

saver_sp = pd.DataFrame(sp_refined,columns=["year","month","day","hour","gen","S_irradiation","H_irradiation","surface_temp","surround_temp","maxgen"])
saver_sp.to_csv("./data/pv_2015_2016_sp_processed.csv",index=False)

saver_ya = pd.DataFrame(ya_refined,columns=["year","month","day","hour","gen","S_irradiation","H_irradiation","surface_temp","surround_temp","maxgen"])
saver_ya.to_csv("./data/pv_2015_2016_ya_processed.csv",index=False)













#print(d)
#d = np.array(d).astype(float)

'''
d2 = np.array(d[0]).reshape(-1,d[0].shape[0])
for i in range(d.shape[0]):
    timeDif = int(d[i,0])%10000 - int(d2[-1,0])%10000
    if( timeDif == 15 or timeDif == 55 or timeDif ==-2345):
        d2 = np.append(d2,d[i].reshape(-1,d[i].shape[0]),axis=0)
print(d2)

for m in range(13):
    count = 0
    for i in d2:
        if (math.floor(i[0] / 1000000) == 2016 * 100 + m):
            count += 1
    print(m, count)


#시간을 일 시 분 으로 쪼갬
d2 = np.append(d2[:,0].reshape(-1,1),d2, axis=1)
d2 = np.append(d2[:,0].reshape(-1,1),d2, axis=1)
d2[:,0] = np.floor(d2[:,0]%1000000/10000)#day
d2[:,1] = np.floor(d2[:,1]%10000/100)#hour
d2[:,2] = np.floor(d2[:,2]%100/1)#minute

print(d)

save = pd.DataFrame(d2,columns=["day","hour","minute","gen","H_irradiation","S_irradiation"])
save.to_csv("./data/pv_2016_gs_processed.csv",index=False)
'''