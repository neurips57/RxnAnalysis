import numpy as np

global_500_res = []
global_val_res = []

data_id = 4
train_size = 518

for split in range(10):
    path = './results/results_%d_%d_%d.txt'%(data_id, split, train_size)
    full_val_data = []
    full_test_data = []
    validation_loss = []
    test_loss = []

    with open(path) as f:
        i = 0
        for line in f:
            if i == 1513:
                x = line.strip().split(" MAE: ")
                y_mae = x[1].strip()[:5].strip()
                x = line.strip().split(" RMSE: ")
                y_rmse = x[1].strip()[:5].strip()
                x = line.strip().split(" R2: ")
                y_r2 = x[1].strip()[:5].strip()
                x = line.strip().split(" Spearman: ")
                y_sprmn = x[1].strip()[:5].strip()
                global_500_res.append([float(y_mae), float(y_rmse), float(y_r2)])

            if i >= 1510:
                i += 1
                continue

            if line.startswith('--- validation'):
                x = line.strip().split(" MAE ")  #RMSE, MAE
                y = x[1].strip()[:5].strip()
                full_val_data.append(line)
                validation_loss.append(float(y))
            
            elif line.startswith('--- test'):
                x = line.strip().split(" MAE ")
                y_mae = x[1].strip()[:5].strip()
                x = line.strip().split(" RMSE ")
                y_rmse = x[1].strip()[:5].strip()
                x = line.strip().split(" R2 ")
                y_r2 = x[1].strip()[:5].strip()
                full_test_data.append(line)
                test_loss.append([float(y_mae), float(y_rmse), float(y_r2)])
            
            i += 1

            

    idx = np.argmin(np.array(validation_loss))
    result = test_loss[idx]
    global_val_res.append(result)

errs = ["MAE", "RMSE", "R2"]

global_500_res = np.array(global_500_res)
print("----------------500 iteration result----------------")
for i in range(3):
    tmp = global_500_res[:,i]
    print(errs[i])
    print(*tmp, sep=", ")
    print("AVG ",round(np.mean(tmp),3))
    print("STD ", round(np.std(tmp),3))

global_val_res = np.array(global_val_res)
print("----------------Val result----------------")
for i in range(3):
    tmp = global_val_res[:,i]
    print(errs[i])
    print(*tmp, sep=", ")
    print("AVG ",round(np.mean(tmp),3))
    print("STD ", round(np.std(tmp),3))




