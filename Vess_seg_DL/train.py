import os, sys
import ConfigParser

config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))


name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '

# making a directory if not present 

result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
    print "Dir already existing"
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)


print "copy the configuration file in the results folder"
if sys.platform=='win32':
    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

