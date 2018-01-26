"""
sumarize.py
"""
def main():
    f = open("summary.txt","w+")
    f1 = open("tweet_data.txt","r")
    f2 = open("cluster.txt","r")
    f3 = open("classify_data.txt","r")
    
    line=f1.readline()
    #print(line)
    while(line !=''):
        f.write(line)
        line=f1.readline()
        
    line=f2.readline()
    while(line !=''):
        f.write(line)
        line=f2.readline()
        
    line=f3.readline()
    while(line !=''):
        f.write(line)
        line=f3.readline()
        
    f.close()
    f1.close()
    f2.close()
    f3.close()
    
    

if __name__ == '__main__':
    main()
