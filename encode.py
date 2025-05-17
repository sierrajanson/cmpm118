import scipy.io

# DATA_FOLDER = "C:\Users\janso\Downloads\P1"
DATA_PATH = r"C:\\Users\\janso\\Downloads\\P1\\WS_P1_S9.mat"

mat_data = scipy.io.loadmat(DATA_PATH)
var = mat_data['ws'][0][0] # ws list basically
print(type(var))
print(len(var))

identifier = None
participant_num = None
series = None
data = None
for i, ele in enumerate(var):
    minivar = ele[0] # values in ws struct
    if i == 1:
        identifier = minivar
    elif i == 2:
        participant_num = minivar 
    elif i == 3:
        series = minivar
    else: # all of data
        data = minivar

    
    # print(i)