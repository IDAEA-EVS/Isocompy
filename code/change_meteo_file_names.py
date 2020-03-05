#change names
import os
import re
def file_name_change_pysplit():
    addre=input("enter folder address to rename to pysplit format:")
    for root, dirs, files in os.walk(addre):
        month_nums=["01","02","03","04","05","06","07","08","09","10","11","12"]
        month_str=["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        for mon in range(0,len(month_nums)):
            match_phrase="^RP...."+month_nums[mon]+".gbl"
            for file in  files:
                if re.match(match_phrase, file):
                    yr=file[2:6]
                    new_name=str(addre+"\RP"+month_str[mon]+yr+".gbl")
                    old_name=str(addre+"\\"+file)
                    #print (old_name,new_name)
                    os.rename(old_name , new_name)
    print ("rename finished")