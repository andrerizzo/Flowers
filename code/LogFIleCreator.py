# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:47:09 2021

@author: andre.rizzo
"""


def Create_Log_File():
    
    from datetime import date
    from sys import path
    
    path.append("/Temp/")
    today = date.today().strftime("%Y_%m_%d")
    filename = "Log_" + today + ".txt"
    
    try:
        f = open(filename, "a")
        f.write("Now the file has more content!\n")

        
    except FileNotFoundError():
        print("File doesn't exist")
        
    finally:
        f.close()


Create_Log_File()
