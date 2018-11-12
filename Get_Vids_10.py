import sys

# sport = str(sys.argv[1])
number = int(sys.argv[1])
# file = open('train_partition.txt', 'r')
# flag = 0
# count = 0

to_use=[8,24,78,388,368,187,469,134,206,55]
'''
8 -> Cycling
24 -> Figure Skating
78 -> Skiing
388 -> Indoor soccer
368 -> Basketball
187 -> Surfing
469 -> Kayaking
134 -> golf
206 ->sprint
55 -> Tennis
'''
for sport in to_use:#range(1,488):
	n=0
	file = open('test_partition.txt', 'r')
	flag = 0
	count = 0
	for line in file:
		split_file = line.split()
		#
		# print split_file
		# print split_file[1]
		# print sport
		string =split_file[1].split(',')
		
		# print string
		# if int(split_file[1]) == sport:
		# 	print 'yes'
		# else:
		# 	print 'no'

		if int(string[0]) == sport:
			print 'youtube-dl '+split_file[0]+' --max-filesize 10m --min-filesize 1m '+' -o '+'Sport_'+split_file[1]+'_vid_'+str(count)+'.mp4'
			count +=1
			if count == number: break
				
