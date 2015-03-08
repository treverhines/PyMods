#!/usr/bin/env python
import json
import subprocess as sp
from misc import timestamp
from misc import divide_list

DEFAULT_PARAMETERS = {
  "N":"default_name",
  "l":"nodes=1:ppn=1,pmem=1000mb,qos=flux,walltime=1:00:00",
  "q":"flux",
  "A":"ehetland_flux",
  "V":"",
  "M":"hinest@umich.edu",
  "e":"log.err",
  "o":"log.out"
  }


VALID_KEYS = ['a','A','b','c','C','d','D','e','f','h','I','j','k','l','m','M',
              'n','N','o','p','P','q','r','S','t','T','u','v','V','w','W','x',
              'X','z']

def chain(file_names,chain_no):
  '''       
  takes a list of PBS files, splits them up into groups, and then    
  adds a line to the end of each PBS file which submits the next file in   
  the group         
  '''
  PBS_chains = divide_list(file_names,chain_no)
  for itr1,PBS_chain in enumerate(PBS_chains):
    for itr2 in range(len(PBS_chain) - 1):
      f = open(PBS_chain[itr2],'a')
      f.write('qsub %s\n' % PBS_chain[itr2+1])
      f.close()
  chain_heads = [i[0] for i in PBS_chains]
  return chain_heads

##------------------------------------------------------------------------------
def _write_header(**kwargs):
  '''
  PARAMETERS
  ----------
    default_parameter_file: json file which contains default PBS parameters
    **kwargs: key word arguments which have PBS parameter names as the key and 
              their corresponding values as the values

  RETURNS
  -------
    header: PBS header

  '''    
  # load parameters and update with kwargs
  parameters = DEFAULT_PARAMETERS
  parameters.update(kwargs)

  header = '#!/bin/sh\n'
  for key,value in parameters.iteritems():
    if key in VALID_KEYS:
      header += '#PBS -%s %s\n' % (key,value)
  return header

##------------------------------------------------------------------------------
class PBSFile(file):
  '''
  PBS file
  '''
  def __init__(self,*args,**kwargs):
    '''
    PARAMETERS
    ----------
      *args: arguments which are nomally given for file initiation
      **kwargs: key word arguments passed to write_header in the event that the
                file is opened with write permission
    '''                 
    file.__init__(self,*args)
    if self.mode == 'w':
      self.write(_write_header(**kwargs))

  def submit(self):
    self.close()
    out = sp.call('qsub %s' % self.name,shell = True)    


