def answer(arr,e_d):    
    s = ""
    for i,v in enumerate(arr):
        s = s + str(e_d[i]) + " : " + str(round(v*100,2)) + "%" + '\n'
    return s
        