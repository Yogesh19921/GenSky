import sys
import os
import Multi_oop as M

if __name__ == "__main__":
    threads = []

    files = os.listdir("datasets/")
    threadID = 1

    for file in files:
        thread = M.MultiObjGP("datasets/" + file)
        thread.start()
        threads.append(thread)
        threadID = threadID + 1

    for t in threads:
        t.join()
