# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(0,10):
        print(i)
    print_hi('PyCharm')
    a = np.random.random((10,1))-0.5
    b = np.random.rand(10,1)-0.5
    print(a)
    print(b)
    print(a.shape)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
