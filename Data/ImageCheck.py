import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Data import *

if __name__ == "__main__":
    count = 0
    image = NiftiData(SWIN_size=(96, 96, 96))
    #print(NiftiData.get_sample(image, 1))
    for item in image:
        #print(item)
        count+=1
    print(f"\n\n Final count: {count} paired images!\n\n")
