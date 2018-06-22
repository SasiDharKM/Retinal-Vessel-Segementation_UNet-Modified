import os, sys
import ConfigParser

config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))


name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')