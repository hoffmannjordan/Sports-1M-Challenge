import sys

# sport = str(sys.argv[1])
number = int(sys.argv[1])
# file = open('train_partition.txt', 'r')
# flag = 0
# count = 0

for sport in range(1,488):
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
			print 'youtube-dl '+split_file[0]+' --max-filesize 10m '+' -o '+'Sport_'+split_file[1]+'_vid_'+str(count)+'.mp4'
			count +=1
			if count == number: break
				
