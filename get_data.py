import numpy as np
import csv
import pandas as pd

def get_data(sport):
	if sport == 'basketball':
		f = pd.io.parsers.read_csv('all_basketball_data.csv')
		data = f.as_matrix()
		#print np.unique(data.T[0])
		games = np.unique(data.T[0])
		ALL_DATA = []#np.empty(len(games))
		#print len(games)
		for j in xrange(len(games)):
			DATA = []
			game_id = games[j]
			for i in xrange(len(data)):
				if data[i][0] == game_id:
					break
			print 'Starting at: ',i
			string=''
			#print list(data[i][-1])[0]#[3]#.split('[')[1].split(']')[0].split(" ")
			#exit()
			game = data[i][0]
			while game == game_id:
				string=''
				if list(data[i][-1])[0] == '[':
					team_1,team_2 = string.join(list(data[i][0])[-6:][0:3]),string.join(list(data[i][0])[-6:][3:])
					#print team_1, team_2
					h,m,s = data[i][2].split(':')
					time = float(m)+float(s)/60.0
					#print time
					if len(data[i][3].split('[')[1].split(']')[0].split(" ")) == 2:
						scorer,score = data[i][3].split('[')[1].split(']')[0].split(" ")
						t1s,t2s = score.split('-')
						#print scorer
						#print t1s,t2s
						if scorer == team_1:
							DATA.append([team_1,team_2,time, float(t1s) , float(t2s)])
						else:
							DATA.append([team_1,team_2,time, float(t2s) , float(t1s)])
				if i != 543238-1:
					i+=1
					game = data[i][0]
				else:
					break
			ALL_DATA.append(DATA)#[j] = DATA
	elif sport == 'football':
		f = pd.io.parsers.read_csv('nfl_data_2.csv')
		data = f.as_matrix()
		ALL_DATA = []
		tmp_data = []
		for i in xrange(len(data)-1):
			print data[i]
			exit()
			if (data[i][0]==data[i+1][0]) and ((data[i][1] == data[i+1][2]) or (data[i][2] == data[i+1][1]) or (data[i][1] == data[i+1][1]) or (data[i][2] == data[i+1][2])):		
			#print data[0]
				team1,team2,quarter,time,score1,score2 = data[i][1],data[i][2],data[i][3],data[i][4],data[i][9],data[i][10]
				#print team1
				#print team2
				print quarter
				print time
				m,s,ms = time.split(':')
				time2=60-15*(float(quarter)) + float(m)+float(s)/60
				# print time2
# 				exit()
				print score1
				print score2
				tmp_data.append([team1,team2,time2,float(score1),float(score2)])
			else:
				ALL_DATA.append(tmp_data)
				tmp_data = []
		#print tmp_data
		#exit()
	
	print ALL_DATA[10]
			
		
		

if __name__=='__main__':
	print get_data('football')