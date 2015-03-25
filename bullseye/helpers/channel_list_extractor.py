import re
def channel_range(s):
    sT = s.strip()
    if re.match('[0-9]+:[0-9]+(~[0-9]+)?(,[0-9]+(~[0-9]+)?)*$',sT) != None:
	spw,selections = s.split(':')
	ranges = selections.split(',')
	channels = []
	for r in ranges:
	  if re.match('[0-9]+~[0-9]+$',r) != None:
	    start,finish = r.split('~')
	    channels = channels + range(int(start),int(finish)+1)
	  else:
	    channels.append(int(r))
	return int(spw),sorted(set(channels))
    else:
	raise argparse.ArgumentTypeError("Channel ranges should be specified as 'spw index':'comma seperated ranges of channels', for example 0:0,2~5,7 will select channels 0,2,3,4,5,7 from spw 0")