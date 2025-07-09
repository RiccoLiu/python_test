#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import threading
import time
import queue

def new_task(delay_second : int):
    while (delay_second >= 0):
        print(f'---- new_task delay_second = {delay_second} ----')
        time.sleep(1)
        delay_second -= 1

class new_task_class(threading.Thread):
    def run(self):
        num = 3

        while (num >= 0):
            print(f'---- new_task_class run num = {num} ----')
            time.sleep(1)
            num -= 1


counter = 0
lock = threading.Lock()

def increment_counter():
    global counter
    with lock:
        for _ in range(10000000):
            counter += 1

msg = queue.Queue()

def producer():
    for i in range(5):
        time.sleep(1)
        msg.put(i)
        print(f"Produced: {i}")

def consumer():
    while True:
        time.sleep(2)
        item = msg.get()
        if item == None:
            break

        print(f"Consumed: {item}")

if __name__ == '__main__':
    '''
        多线程:
            创建线程：

            线程池：

            线程同步：
                threading.Lock()

            线程通信：
                queue.Queue()  
            
            守护线程：
                主程序结束后，线程会强制结束，守护线程不会阻塞主线程，类似 C++ 的 detach 线程。

                thread.daemon = True  # 设置为守护线程 
    '''

    # 创建线程
    thread = threading.Thread(target=new_task, args=(3,) )
    # thread = threading.Thread(target=new_task, kwargs={'delay_second':2} )
    thread.start()

    # 创建线程2
    thread2 = new_task_class()
    thread2.start()

    thread.join()
    thread2.join()

    print('-----------------')

    # 线程池
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for i in range(5):
            executor.submit(new_task, i)

    print('-----------------')

    # 线程同步
    lock_thread1 = threading.Thread(target=increment_counter)
    lock_thread2 = threading.Thread(target=increment_counter)

    lock_thread1.start()
    lock_thread2.start()

    lock_thread1.join()
    lock_thread2.join()

    print("counter:", counter)

    print('-----------------')

    # 线程通信 queue
    msg_thread1 = threading.Thread(target=producer)
    msg_thread2 = threading.Thread(target=consumer)

    msg_thread1.start()
    msg_thread2.start()

    msg_thread1.join()

    msg.put(None)

    msg_thread2.join()
